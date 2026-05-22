import argparse
import math
import random
import time

import mujoco

from golf_3joint_common import (
    BASELINE_CANDIDATE,
    EPISODE_STEPS,
    MAX_ELBOW_CTRL,
    MAX_SHOULDER_CTRL,
    MAX_WRIST_CTRL,
    apply_three_joint_controls,
    ball_x as INITIAL_BALL_X,
    clip,
    make_single_arm_model,
    print_candidate,
    print_copy_paste_controls,
)


DEFAULT_TRIALS = 10000
INVALID_SWING_REWARD = -100000.0
MIN_FORWARD_CLUB_SPEED = 0.50
MIN_FORWARD_BALL_SPEED = 0.20
MAX_COLLAPSE_HIT_STEP_BUFFER = 20

TAKEAWAY_SAMPLE_FRACTION = 0.55
TOP_UPPER_TARGET = math.radians(82.0)
TOP_FOREARM_TARGET = math.radians(12.0)
TOP_CLUB_TARGET = math.radians(88.0)
IMPACT_LINE_TARGET = 0.0


def random_near(rng, base, spread, min_value, max_value):
    return clip(rng.uniform(base - spread, base + spread), min_value, max_value)


def make_candidate(rng):
    takeaway_end = rng.randint(120, 260)
    shoulder_fire_end = rng.randint(takeaway_end + 45, takeaway_end + 150)
    release_end = rng.randint(shoulder_fire_end + 35, shoulder_fire_end + 135)

    return {
        "t1": takeaway_end,
        "t2": shoulder_fire_end,
        "t3": release_end,
        # Takeaway: all three joints hinge away from the ball together.
        "s1": random_near(rng, 6.5, 3.5, 0.0, MAX_SHOULDER_CTRL),
        "e1": random_near(rng, 1.8, 2.2, 0.0, MAX_ELBOW_CTRL),
        "w1": random_near(rng, 3.0, 1.5, 0.0, MAX_WRIST_CTRL),
        # Early downswing: shoulder fires first while the elbow/wrist are quieter.
        "s2": random_near(rng, -8.0, 2.0, -MAX_SHOULDER_CTRL, -2.0),
        "e2": random_near(rng, -0.8, 1.2, -MAX_ELBOW_CTRL, 1.5),
        "w2": random_near(rng, -0.6, 1.0, -MAX_WRIST_CTRL, 1.0),
        # Release: elbow and wrist join the shoulder before impact.
        "s3": random_near(rng, -7.0, 3.0, -MAX_SHOULDER_CTRL, 0.0),
        "e3": random_near(rng, -3.0, 1.2, -MAX_ELBOW_CTRL, 0.0),
        "w3": random_near(rng, -3.4, 0.8, -MAX_WRIST_CTRL, -0.8),
    }


def contact_includes(data, geom_a, geom_b):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
            return True
    return False


def angle_score(angle, target, tolerance):
    error = abs(angle - target)
    return max(0.0, 1.0 - error / tolerance)


def joint_angles(data):
    shoulder = data.qpos[0]
    elbow = data.qpos[1]
    wrist = data.qpos[2]
    upper_abs = shoulder
    forearm_abs = shoulder + elbow
    club_abs = shoulder + elbow + wrist
    return shoulder, elbow, wrist, upper_abs, forearm_abs, club_abs


def sequence_score(samples, candidate):
    takeaway = samples.get("takeaway")
    top = samples.get("top")
    impact = samples.get("impact")

    score = 0.0
    if takeaway:
        shoulder, elbow, wrist, upper_abs, forearm_abs, club_abs = takeaway
        if shoulder > 0 and elbow > 0 and wrist > 0:
            score += 1.0
        if upper_abs > 0 and forearm_abs > 0 and club_abs > 0:
            score += 1.0

    if top:
        _, _, wrist, upper_abs, forearm_abs, club_abs = top
        score += angle_score(upper_abs, TOP_UPPER_TARGET, math.radians(45.0))
        score += angle_score(forearm_abs, TOP_FOREARM_TARGET, math.radians(55.0))
        score += angle_score(club_abs, TOP_CLUB_TARGET, math.radians(45.0))
        score += angle_score(wrist, 0.0, math.radians(55.0))

    if candidate["s2"] < candidate["e2"] and candidate["s2"] < candidate["w2"]:
        score += 1.0
    if abs(candidate["e2"]) < abs(candidate["e3"]) and abs(candidate["w2"]) < abs(candidate["w3"]):
        score += 1.0

    if impact:
        _, _, _, upper_abs, forearm_abs, club_abs = impact
        score += angle_score(upper_abs, IMPACT_LINE_TARGET, math.radians(35.0))
        score += angle_score(forearm_abs, IMPACT_LINE_TARGET, math.radians(35.0))
        score += angle_score(club_abs, IMPACT_LINE_TARGET, math.radians(35.0))

    return score


def simulate_swing(model, candidate, require_hit=True):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

    max_ball_x = INITIAL_BALL_X
    initial_ball_z = data.xpos[ball_id][2]
    hit_ball = False
    valid_impact = False
    first_hit_step = None
    min_tip_to_ball = math.inf
    max_ball_z = data.xpos[ball_id][2]
    impact_club_vx = 0.0
    impact_ball_vx = 0.0
    impact_ball_vz = 0.0
    samples = {}
    takeaway_sample_step = int(candidate["t1"] * TAKEAWAY_SAMPLE_FRACTION)
    previous_tip_pos = data.site_xpos[club_tip_id].copy()
    previous_ball_pos = data.xpos[ball_id].copy()

    for step in range(EPISODE_STEPS):
        apply_three_joint_controls(data, candidate, step)
        mujoco.mj_step(model, data)

        ball_pos = data.xpos[ball_id]
        tip_pos = data.site_xpos[club_tip_id]
        club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
        ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep
        tip_to_ball = math.dist(tip_pos, ball_pos)
        min_tip_to_ball = min(min_tip_to_ball, tip_to_ball)
        max_ball_z = max(max_ball_z, ball_pos[2])

        if step == takeaway_sample_step:
            samples["takeaway"] = joint_angles(data)
        if step == candidate["t1"]:
            samples["top"] = joint_angles(data)

        if contact_includes(data, club_head_geom_id, ball_geom_id):
            hit_ball = True
            if first_hit_step is None:
                first_hit_step = step
                samples["impact"] = joint_angles(data)
                impact_club_vx = club_velocity[0]
                impact_ball_vx = ball_velocity[0]
                impact_ball_vz = ball_velocity[2]

                active_downswing = candidate["t1"] <= step <= candidate["t3"] + MAX_COLLAPSE_HIT_STEP_BUFFER
                moving_forward = impact_club_vx >= MIN_FORWARD_CLUB_SPEED
                ball_launched_forward = impact_ball_vx >= MIN_FORWARD_BALL_SPEED
                valid_impact = active_downswing and moving_forward and ball_launched_forward

        if ball_pos[0] > max_ball_x:
            max_ball_x = ball_pos[0]

        previous_tip_pos = tip_pos.copy()
        previous_ball_pos = ball_pos.copy()

    distance = max_ball_x - INITIAL_BALL_X
    height_gain = max(0.0, max_ball_z - initial_ball_z)
    motion_score = sequence_score(samples, candidate)
    reward = (
        distance * 100.0
        + height_gain * 40.0
        + max(0.0, impact_ball_vx) * 2.0
        + max(0.0, impact_ball_vz)
        + motion_score * 8.0
    )
    if require_hit and not valid_impact:
        reward = INVALID_SWING_REWARD + motion_score - min_tip_to_ball

    return {
        "reward": reward,
        "distance": distance,
        "height_gain": height_gain,
        "hit_ball": hit_ball,
        "valid_impact": valid_impact,
        "first_hit_step": first_hit_step,
        "impact_club_vx": impact_club_vx,
        "impact_ball_vx": impact_ball_vx,
        "impact_ball_vz": impact_ball_vz,
        "sequence_score": motion_score,
        "min_tip_to_ball": min_tip_to_ball,
    }


def search(trials, seed=None, progress_every=100):
    rng = random.Random(seed)
    model = make_single_arm_model()

    baseline_result = simulate_swing(model, BASELINE_CANDIDATE, require_hit=False)
    print("Hardcoded baseline distance:", round(baseline_result["distance"], 4))
    print("Hardcoded baseline hit ball:", baseline_result["hit_ball"])
    print("Hardcoded baseline valid impact:", baseline_result["valid_impact"])
    print("Hardcoded baseline min tip-to-ball:", round(baseline_result["min_tip_to_ball"], 4))

    best_result = simulate_swing(model, BASELINE_CANDIDATE, require_hit=True)
    best_candidate = dict(BASELINE_CANDIDATE)
    best_valid_result = None
    best_valid_candidate = None
    start_time = time.time()

    for trial in range(1, trials + 1):
        candidate = make_candidate(rng)
        result = simulate_swing(model, candidate, require_hit=True)
        if result["valid_impact"] and (
            best_valid_result is None or result["reward"] > best_valid_result["reward"]
        ):
            best_valid_result = result
            best_valid_candidate = candidate

        if result["reward"] > best_result["reward"]:
            best_result = result
            best_candidate = candidate
            print(
                "New best at trial",
                trial,
                "distance",
                round(best_result["distance"], 4),
                "height",
                round(best_result["height_gain"], 4),
                "hit",
                best_result["hit_ball"],
                "valid",
                best_result["valid_impact"],
                "first_hit_step",
                best_result["first_hit_step"],
                "sequence",
                round(best_result["sequence_score"], 2),
            )
            print_candidate(best_candidate)

        if progress_every and trial % progress_every == 0:
            elapsed = time.time() - start_time
            print(
                "Trial",
                trial,
                "of",
                trials,
                "best distance",
                round(best_result["distance"], 4),
                "height",
                round(best_result["height_gain"], 4),
                "hit",
                best_result["hit_ball"],
                "valid",
                best_result["valid_impact"],
                "valid_found",
                best_valid_result is not None,
                "elapsed_sec",
                round(elapsed, 1),
            )

    final_result = best_valid_result if best_valid_result is not None else best_result
    final_candidate = best_valid_candidate if best_valid_candidate is not None else best_candidate

    print("Training complete")
    if best_valid_result is None:
        print("No valid downswing impact found. Do not paste this candidate into the viewer yet.")
    print("Final best reward:", round(final_result["reward"], 4))
    print("Final best distance:", round(final_result["distance"], 4))
    print("Final best height gain:", round(final_result["height_gain"], 4))
    print("Hit ball:", final_result["hit_ball"])
    print("Valid downswing impact:", final_result["valid_impact"])
    print("First hit step:", final_result["first_hit_step"])
    print("Impact club vx:", round(final_result["impact_club_vx"], 4))
    print("Impact ball vx:", round(final_result["impact_ball_vx"], 4))
    print("Impact ball vz:", round(final_result["impact_ball_vz"], 4))
    print("Sequence score:", round(final_result["sequence_score"], 4))
    print("Minimum tip-to-ball distance:", round(final_result["min_tip_to_ball"], 4))
    print_candidate(final_candidate)
    if final_result["valid_impact"]:
        print_copy_paste_controls(final_candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="Random-search 3-joint golf swing controls.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Number of random candidates to test.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for repeatable training.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N trials.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search(args.trials, seed=args.seed, progress_every=args.progress_every)