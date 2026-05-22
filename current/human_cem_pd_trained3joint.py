import argparse
import math
import random
import time

import mujoco

import golf_3joint_common
from golf_3joint_common import (
    EPISODE_STEPS,
    MAX_ELBOW_CTRL,
    MAX_SHOULDER_CTRL,
    MAX_WRIST_CTRL,
    make_single_arm_model,
)


INVALID_SWING_REWARD = -100000.0
MIN_FORWARD_CLUB_SPEED = 0.50
MIN_FORWARD_BALL_SPEED = 0.20
POST_IMPACT_WINDOW_STEPS = 20
MIN_POST_IMPACT_DISTANCE = 0.03

KP = (16.0, 11.0, 9.0)
KD = (2.4, 1.7, 1.4)
CTRL_LIMITS = (MAX_SHOULDER_CTRL, MAX_ELBOW_CTRL, MAX_WRIST_CTRL)

PARAMS = [
    ("top_step", 80.0, 260.0),
    ("impact_gap", 70.0, 280.0),
    ("finish_gap", 40.0, 260.0),
    ("elbow_lag", 0.0, 90.0),
    ("wrist_lag", 5.0, 130.0),
    ("top_s", 0.20, 1.90),
    ("top_e", -1.40, 1.20),
    ("top_w", -1.00, 1.80),
    ("impact_s", -0.45, 0.45),
    ("impact_e", -0.45, 0.45),
    ("impact_w", -0.45, 0.45),
    ("finish_s", -1.40, 0.60),
    ("finish_e", -1.20, 0.80),
    ("finish_w", -1.20, 0.80),
]

INITIAL_MEAN = [
    150.0,
    170.0,
    130.0,
    25.0,
    55.0,
    1.15,
    -0.35,
    0.75,
    0.02,
    -0.02,
    0.02,
    -0.75,
    -0.35,
    -0.30,
]

INITIAL_STD = [
    42.0,
    55.0,
    55.0,
    20.0,
    24.0,
    0.45,
    0.55,
    0.55,
    0.20,
    0.20,
    0.20,
    0.45,
    0.45,
    0.45,
]

MIN_STD = [
    5.0,
    6.0,
    6.0,
    4.0,
    4.0,
    0.05,
    0.05,
    0.05,
    0.03,
    0.03,
    0.03,
    0.05,
    0.05,
    0.05,
]


def clip(value, low, high):
    return max(low, min(high, value))


def smoothstep(x):
    x = clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a + (b - a) * t


def interpolate_pose(start_pose, end_pose, t):
    t = smoothstep(t)
    return tuple(lerp(start_pose[i], end_pose[i], t) for i in range(3))


def sample_vector(rng, mean, std):
    vector = []
    for i, (_, low, high) in enumerate(PARAMS):
        vector.append(clip(rng.gauss(mean[i], std[i]), low, high))
    return vector


def vector_to_candidate(vector):
    top_step = int(round(vector[0]))
    impact_step = int(round(top_step + vector[1]))
    finish_step = int(round(impact_step + vector[2]))
    elbow_lag = int(round(vector[3]))
    wrist_lag = int(round(max(vector[4], elbow_lag + 5)))
    wrist_lag = min(wrist_lag, max(5, impact_step - top_step - 10))
    elbow_lag = min(elbow_lag, max(0, wrist_lag - 5))

    return {
        "top_step": top_step,
        "impact_step": impact_step,
        "finish_step": finish_step,
        "elbow_lag": elbow_lag,
        "wrist_lag": wrist_lag,
        "top_pose": (vector[5], vector[6], vector[7]),
        "impact_pose": (vector[8], vector[9], vector[10]),
        "finish_pose": (vector[11], vector[12], vector[13]),
    }


def candidate_to_vector(candidate):
    return [
        float(candidate["top_step"]),
        float(candidate["impact_step"] - candidate["top_step"]),
        float(candidate["finish_step"] - candidate["impact_step"]),
        float(candidate["elbow_lag"]),
        float(candidate["wrist_lag"]),
        *candidate["top_pose"],
        *candidate["impact_pose"],
        *candidate["finish_pose"],
    ]


def target_angles(candidate, step):
    address = (0.0, 0.0, 0.0)
    top = candidate["top_pose"]
    impact = candidate["impact_pose"]
    finish = candidate["finish_pose"]
    top_step = candidate["top_step"]
    impact_step = candidate["impact_step"]
    finish_step = candidate["finish_step"]

    if step < top_step:
        return interpolate_pose(address, top, step / max(1, top_step))

    if step < impact_step:
        target = []
        lags = (0, candidate["elbow_lag"], candidate["wrist_lag"])
        for joint in range(3):
            joint_start = top_step + lags[joint]
            if step <= joint_start:
                target.append(top[joint])
            else:
                t = (step - joint_start) / max(1, impact_step - joint_start)
                target.append(lerp(top[joint], impact[joint], smoothstep(t)))
        return tuple(target)

    if step < finish_step:
        return interpolate_pose(impact, finish, (step - impact_step) / max(1, finish_step - impact_step))

    return finish


def apply_pd_controls(data, candidate, step):
    target = target_angles(candidate, step)
    ctrl_energy = 0.0
    for i in range(3):
        command = KP[i] * (target[i] - data.qpos[i]) - KD[i] * data.qvel[i]
        command = clip(command, -CTRL_LIMITS[i], CTRL_LIMITS[i])
        data.ctrl[i] = command
        ctrl_energy += command * command
    return ctrl_energy


def contact_includes(data, geom_a, geom_b):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
            return True
    return False


def angle_score(angle, target, tolerance):
    return max(0.0, 1.0 - abs(angle - target) / tolerance)


def joint_absolute_angles(data):
    shoulder = data.qpos[0]
    forearm = data.qpos[0] + data.qpos[1]
    club = data.qpos[0] + data.qpos[1] + data.qpos[2]
    return shoulder, forearm, club


def line_score(data):
    shoulder, forearm, club = joint_absolute_angles(data)
    tolerance = math.radians(28.0)
    score = 0.0
    score += angle_score(shoulder - forearm, 0.0, tolerance)
    score += angle_score(forearm - club, 0.0, tolerance)
    score += angle_score(club - shoulder, 0.0, tolerance)
    return score / 3.0


def top_pose_score(data):
    shoulder, forearm, club = joint_absolute_angles(data)
    score = 0.0
    score += angle_score(abs(shoulder), math.radians(65.0), math.radians(45.0))
    score += angle_score(abs(club), math.radians(90.0), math.radians(45.0))
    score += angle_score(abs(club - forearm), math.radians(60.0), math.radians(55.0))
    return score / 3.0


def print_startup_diagnostics(model):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")
    ball_pos = data.xpos[ball_id]
    tip_pos = data.site_xpos[club_tip_id]
    print("Loaded shared file:", golf_3joint_common.__file__)
    print("Configured ball_x:", golf_3joint_common.ball_x)
    print("Initial ball position:", [round(float(v), 4) for v in ball_pos])
    print("Initial club tip position:", [round(float(v), 4) for v in tip_pos])
    print("Initial tip-to-ball distance:", round(math.dist(tip_pos, ball_pos), 4))
    print("Initial contacts:", data.ncon)


def simulate_pd_swing(model, candidate, require_hit=True):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")

    initial_ball_x = data.xpos[ball_id][0]
    initial_ball_z = data.xpos[ball_id][2]
    max_ball_x = initial_ball_x
    max_ball_z = initial_ball_z
    min_tip_to_ball = math.inf
    active_min_tip_to_ball = math.inf
    active_max_club_vx = -math.inf
    active_max_club_vz = -math.inf
    first_hit_step = None
    hit_ball = False
    impact_ball_x = None
    impact_club_vx = 0.0
    impact_ball_vx = 0.0
    impact_ball_vz = 0.0
    post_impact_max_ball_vx = 0.0
    post_impact_max_ball_vz = 0.0
    post_impact_distance = 0.0
    impact_was_active_downswing = False
    impact_club_was_forward = False
    top_score_sample = 0.0
    impact_line_score = 0.0
    total_ctrl_energy = 0.0

    previous_tip_pos = data.site_xpos[club_tip_id].copy()
    previous_ball_pos = data.xpos[ball_id].copy()

    for step in range(EPISODE_STEPS):
        total_ctrl_energy += apply_pd_controls(data, candidate, step)
        mujoco.mj_step(model, data)

        ball_pos = data.xpos[ball_id]
        tip_pos = data.site_xpos[club_tip_id]
        club_velocity = (tip_pos - previous_tip_pos) / model.opt.timestep
        ball_velocity = (ball_pos - previous_ball_pos) / model.opt.timestep
        tip_to_ball = math.dist(tip_pos, ball_pos)
        active_downswing = candidate["top_step"] <= step <= candidate["impact_step"] + 35

        min_tip_to_ball = min(min_tip_to_ball, tip_to_ball)
        max_ball_x = max(max_ball_x, ball_pos[0])
        max_ball_z = max(max_ball_z, ball_pos[2])
        if active_downswing:
            active_min_tip_to_ball = min(active_min_tip_to_ball, tip_to_ball)
            active_max_club_vx = max(active_max_club_vx, club_velocity[0])
            active_max_club_vz = max(active_max_club_vz, club_velocity[2])

        if step == candidate["top_step"]:
            top_score_sample = top_pose_score(data)

        if contact_includes(data, club_head_geom_id, ball_geom_id):
            hit_ball = True
            if first_hit_step is None:
                first_hit_step = step
                impact_ball_x = ball_pos[0]
                impact_club_vx = club_velocity[0]
                impact_ball_vx = ball_velocity[0]
                impact_ball_vz = ball_velocity[2]
                impact_was_active_downswing = active_downswing
                impact_club_was_forward = impact_club_vx >= MIN_FORWARD_CLUB_SPEED
                impact_line_score = line_score(data)

        if first_hit_step is not None and step <= first_hit_step + POST_IMPACT_WINDOW_STEPS:
            post_impact_max_ball_vx = max(post_impact_max_ball_vx, ball_velocity[0])
            post_impact_max_ball_vz = max(post_impact_max_ball_vz, ball_velocity[2])
            if impact_ball_x is not None:
                post_impact_distance = max(post_impact_distance, ball_pos[0] - impact_ball_x)

        previous_tip_pos = tip_pos.copy()
        previous_ball_pos = ball_pos.copy()

    if active_min_tip_to_ball == math.inf:
        active_min_tip_to_ball = min_tip_to_ball
    if active_max_club_vx == -math.inf:
        active_max_club_vx = 0.0
    if active_max_club_vz == -math.inf:
        active_max_club_vz = 0.0

    ball_launched_forward = (
        post_impact_max_ball_vx >= MIN_FORWARD_BALL_SPEED
        or post_impact_distance >= MIN_POST_IMPACT_DISTANCE
    )
    valid_impact = impact_was_active_downswing and impact_club_was_forward and ball_launched_forward

    distance = max_ball_x - initial_ball_x
    height_gain = max(0.0, max_ball_z - initial_ball_z)
    sequence_score = 0.0
    if candidate["elbow_lag"] >= 0 and candidate["wrist_lag"] > candidate["elbow_lag"]:
        sequence_score += 1.0
    sequence_score += top_score_sample
    sequence_score += impact_line_score

    reward = (
        distance * 120.0
        + height_gain * 120.0
        + max(0.0, post_impact_max_ball_vx) * 6.0
        + max(0.0, post_impact_max_ball_vz) * 12.0
        + top_score_sample * 25.0
        + impact_line_score * 45.0
        + sequence_score * 20.0
        - total_ctrl_energy * 0.00002
    )
    if require_hit and not valid_impact:
        reward = (
            INVALID_SWING_REWARD
            - active_min_tip_to_ball * 300.0
            + max(0.0, active_max_club_vx) * 8.0
            + top_score_sample * 25.0
            + sequence_score * 20.0
        )

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
        "post_impact_max_ball_vx": post_impact_max_ball_vx,
        "post_impact_max_ball_vz": post_impact_max_ball_vz,
        "post_impact_distance": post_impact_distance,
        "sequence_score": sequence_score,
        "top_score": top_score_sample,
        "impact_line_score": impact_line_score,
        "min_tip_to_ball": min_tip_to_ball,
        "active_min_tip_to_ball": active_min_tip_to_ball,
        "active_max_club_vx": active_max_club_vx,
        "active_max_club_vz": active_max_club_vz,
    }


def mean(values):
    return sum(values) / len(values)


def stddev(values, center):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def update_distribution(elite_vectors, old_mean, old_std, smoothing):
    new_mean = []
    new_std = []
    for i, (_, low, high) in enumerate(PARAMS):
        values = [vector[i] for vector in elite_vectors]
        elite_mean = clip(mean(values), low, high)
        elite_std = max(stddev(values, elite_mean), MIN_STD[i])
        new_mean.append((1.0 - smoothing) * old_mean[i] + smoothing * elite_mean)
        new_std.append((1.0 - smoothing) * old_std[i] + smoothing * elite_std)
    return new_mean, new_std


def print_result(prefix, result):
    print(
        prefix,
        "reward",
        round(result["reward"], 3),
        "distance",
        round(result["distance"], 4),
        "height",
        round(result["height_gain"], 4),
        "hit",
        result["hit_ball"],
        "valid",
        result["valid_impact"],
        "first_hit_step",
        result["first_hit_step"],
        "post_vx",
        round(result["post_impact_max_ball_vx"], 4),
        "post_vz",
        round(result["post_impact_max_ball_vz"], 4),
        "line",
        round(result["impact_line_score"], 3),
    )


def print_pd_candidate(candidate):
    print("\n# Paste this dictionary into human_golf_arm3joint_pd.py as PD_SWING_CANDIDATE:\n")
    print("PD_SWING_CANDIDATE = {")
    print(f'    "top_step": {candidate["top_step"]},')
    print(f'    "impact_step": {candidate["impact_step"]},')
    print(f'    "finish_step": {candidate["finish_step"]},')
    print(f'    "elbow_lag": {candidate["elbow_lag"]},')
    print(f'    "wrist_lag": {candidate["wrist_lag"]},')
    print(f'    "top_pose": ({candidate["top_pose"][0]:.4f}, {candidate["top_pose"][1]:.4f}, {candidate["top_pose"][2]:.4f}),')
    print(f'    "impact_pose": ({candidate["impact_pose"][0]:.4f}, {candidate["impact_pose"][1]:.4f}, {candidate["impact_pose"][2]:.4f}),')
    print(f'    "finish_pose": ({candidate["finish_pose"][0]:.4f}, {candidate["finish_pose"][1]:.4f}, {candidate["finish_pose"][2]:.4f}),')
    print("}")


def train(generations, population, elite_count, seed=None, smoothing=0.7):
    rng = random.Random(seed)
    model = make_single_arm_model()
    print_startup_diagnostics(model)

    mean_vector = list(INITIAL_MEAN)
    std_vector = list(INITIAL_STD)
    best_result = None
    best_candidate = None
    best_valid_result = None
    best_valid_candidate = None
    start_time = time.time()

    for generation in range(1, generations + 1):
        scored = []
        for _ in range(population):
            vector = sample_vector(rng, mean_vector, std_vector)
            candidate = vector_to_candidate(vector)
            result = simulate_pd_swing(model, candidate, require_hit=True)
            scored.append((result["reward"], vector, candidate, result))

            if best_result is None or result["reward"] > best_result["reward"]:
                best_result = result
                best_candidate = candidate
            if result["valid_impact"] and (
                best_valid_result is None or result["reward"] > best_valid_result["reward"]
            ):
                best_valid_result = result
                best_valid_candidate = candidate

        scored.sort(key=lambda item: item[0], reverse=True)
        if best_valid_result is not None:
            valid_items = [item for item in scored if item[3]["valid_impact"]]
            elite_source = valid_items if len(valid_items) >= max(2, elite_count // 2) else scored
        else:
            elite_source = scored

        elite_vectors = [item[1] for item in elite_source[:elite_count]]
        mean_vector, std_vector = update_distribution(elite_vectors, mean_vector, std_vector, smoothing)

        elapsed = time.time() - start_time
        print_result(f"Generation {generation}/{generations}", scored[0][3])
        print("valid_found", best_valid_result is not None, "elapsed_sec", round(elapsed, 1))

    final_result = best_valid_result if best_valid_result is not None else best_result
    final_candidate = best_valid_candidate if best_valid_candidate is not None else best_candidate

    print("Training complete")
    if best_valid_result is None:
        print("No valid PD downswing impact found yet.")
    print_result("Final best", final_result)
    print("Impact club vx:", round(final_result["impact_club_vx"], 4))
    print("Post-impact max ball vx:", round(final_result["post_impact_max_ball_vx"], 4))
    print("Post-impact max ball vz:", round(final_result["post_impact_max_ball_vz"], 4))
    print("Post-impact distance:", round(final_result["post_impact_distance"], 4))
    print("Top pose score:", round(final_result["top_score"], 4))
    print("Impact line score:", round(final_result["impact_line_score"], 4))
    print("Sequence score:", round(final_result["sequence_score"], 4))
    print("Active downswing min tip-to-ball:", round(final_result["active_min_tip_to_ball"], 4))
    print("Active downswing max club vx:", round(final_result["active_max_club_vx"], 4))
    print_pd_candidate(final_candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="CEM trainer using smooth target poses plus PD control.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoothing", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        generations=args.generations,
        population=args.population,
        elite_count=args.elite_count,
        seed=args.seed,
        smoothing=args.smoothing,
    )