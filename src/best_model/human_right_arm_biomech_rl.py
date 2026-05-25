import argparse
import math
import random
import time

import mujoco

from golf_3joint_common import CLUB_PRESETS, get_club_launch_profile, normalize_club_name
from human_right_arm_biomech_cem import (
    EPISODE_STEPS,
    MIN_FORWARD_CLUB_SPEED,
    MIN_POST_IMPACT_DISTANCE,
    POST_IMPACT_WINDOW_STEPS,
    apply_pd_controls,
    initial_mean,
    initial_std,
    min_std,
    parameter_specs,
    print_candidate,
    target_angles,
    vector_to_candidate,
)
from human_right_arm_biomech_static import (
    DEFAULT_HAND,
    HAND_SIGNS,
    apply_setup_pose,
    make_static_model,
    normalize_hand,
)


MIN_BACKSWING_ARC = 0.75
MIN_FORWARD_BALL_SPEED = 0.20


def clip(value, low, high):
    return max(low, min(high, value))


def vector_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    length = norm(a)
    if length < 1e-9:
        return (0.0, 1.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


def plane_distance(point, plane_point, plane_normal):
    return abs(dot(vector_sub(point, plane_point), plane_normal))


def contact_includes(data, geom_a, geom_b):
    for i in range(data.ncon):
        contact = data.contact[i]
        if {contact.geom1, contact.geom2} == {geom_a, geom_b}:
            return True
    return False


def plane_score(mean_error):
    return 1.0 / (1.0 + 18.0 * mean_error)


def sample_policy_vector(rng, mean, std, specs):
    vector = []
    eps = []
    for i, (_, low, high) in enumerate(specs):
        noise = rng.gauss(0.0, 1.0)
        eps.append(noise)
        vector.append(clip(mean[i] + std[i] * noise, low, high))
    return vector, eps


def update_policy(mean, std, epsilons, returns, specs, learning_rate, std_decay):
    if len(returns) < 2:
        return mean, std

    return_mean = sum(returns) / len(returns)
    variance = sum((value - return_mean) ** 2 for value in returns) / len(returns)
    return_std = math.sqrt(max(variance, 1e-9))
    normalized_returns = [(value - return_mean) / return_std for value in returns]
    floors = min_std()

    next_mean = []
    next_std = []
    for i, (_, low, high) in enumerate(specs):
        gradient = sum(normalized_returns[j] * epsilons[j][i] for j in range(len(returns)))
        gradient /= len(returns)
        next_mean.append(clip(mean[i] + learning_rate * std[i] * gradient, low, high))
        next_std.append(max(floors[i], std[i] * std_decay))
    return next_mean, next_std


def evaluate_rl_swing(model, candidate, launch_profile):
    data = mujoco.MjData(model)
    apply_setup_pose(model, data, candidate.get("hand", DEFAULT_HAND))

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "golf_ball")
    club_head_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head_geom")
    club_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "club_tip")
    shoulder_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "shoulder_site")

    shoulder = tuple(float(v) for v in data.site_xpos[shoulder_site_id])
    ball_start = tuple(float(v) for v in data.xpos[ball_id])
    target_axis = (1.0, 0.0, 0.0)
    plane_normal = normalize(cross(target_axis, vector_sub(shoulder, ball_start)))

    initial_ball_x = data.xpos[ball_id][0]
    initial_ball_y = data.xpos[ball_id][1]
    initial_ball_z = data.xpos[ball_id][2]
    max_ball_x = initial_ball_x
    max_ball_z = initial_ball_z
    max_offline_y = 0.0
    backswing_arc = 0.0
    backswing_plane_error = 0.0
    backswing_samples = 0
    followthrough_plane_error = 0.0
    followthrough_samples = 0
    first_hit_step = None
    impact_ball_x = None
    impact_club_vx = 0.0
    post_impact_max_ball_vx = 0.0
    post_impact_max_ball_vz = 0.0
    post_impact_distance = 0.0
    hit_ball = False
    valid_impact = False
    total_ctrl_energy = 0.0

    previous_tip = data.site_xpos[club_tip_id].copy()
    previous_ball = data.xpos[ball_id].copy()
    down_start_step = candidate.get(
        "down_start_step",
        candidate["top_step"] + candidate.get("top_hold", 0),
    )

    for step in range(EPISODE_STEPS):
        total_ctrl_energy += apply_pd_controls(data, candidate, step)
        mujoco.mj_step(model, data)

        tip = data.site_xpos[club_tip_id]
        ball = data.xpos[ball_id]
        club_velocity = (tip - previous_tip) / model.opt.timestep
        ball_velocity = (ball - previous_ball) / model.opt.timestep
        step_arc = math.dist(tip, previous_tip)
        tip_plane_error = plane_distance(tip, ball_start, plane_normal)
        active_downswing = down_start_step <= step <= candidate["impact_step"] + 55

        max_ball_x = max(max_ball_x, ball[0])
        max_ball_z = max(max_ball_z, ball[2])
        max_offline_y = max(max_offline_y, abs(ball[1] - initial_ball_y))

        if step <= down_start_step:
            backswing_arc += step_arc
            backswing_plane_error += tip_plane_error
            backswing_samples += 1
        elif step >= candidate["impact_step"] and step <= candidate["finish_step"] + 80:
            followthrough_plane_error += tip_plane_error
            followthrough_samples += 1

        if contact_includes(data, club_head_geom_id, ball_geom_id):
            hit_ball = True
            if first_hit_step is None and active_downswing:
                first_hit_step = step
                impact_ball_x = ball[0]
                impact_club_vx = club_velocity[0]

        if first_hit_step is not None and step <= first_hit_step + POST_IMPACT_WINDOW_STEPS:
            post_impact_max_ball_vx = max(post_impact_max_ball_vx, ball_velocity[0])
            post_impact_max_ball_vz = max(post_impact_max_ball_vz, ball_velocity[2])
            if impact_ball_x is not None:
                post_impact_distance = max(post_impact_distance, ball[0] - impact_ball_x)

        previous_tip = tip.copy()
        previous_ball = ball.copy()

    backswing_plane_mean = backswing_plane_error / max(1, backswing_samples)
    followthrough_plane_mean = followthrough_plane_error / max(1, followthrough_samples)
    backswing_plane_score = plane_score(backswing_plane_mean)
    followthrough_plane_score = plane_score(followthrough_plane_mean)

    ball_launched_forward = (
        post_impact_max_ball_vx >= MIN_FORWARD_BALL_SPEED
        or post_impact_distance >= MIN_POST_IMPACT_DISTANCE
    )
    valid_impact = (
        first_hit_step is not None
        and impact_club_vx >= MIN_FORWARD_CLUB_SPEED
        and ball_launched_forward
    )

    distance = max_ball_x - initial_ball_x
    height_gain = max(0.0, max_ball_z - initial_ball_z)
    short_arc = max(0.0, MIN_BACKSWING_ARC - backswing_arc)
    capped_backswing_arc = min(backswing_arc, 2.20)
    on_plane_arc = capped_backswing_arc * backswing_plane_score
    reward = (
        distance * launch_profile["distance_weight"]
        + height_gain * launch_profile["height_weight"] * 0.35
        + on_plane_arc * 1100.0
        + backswing_plane_score * 520.0
        + followthrough_plane_score * 260.0
        + max(0.0, post_impact_max_ball_vx) * launch_profile["forward_speed_weight"]
        + max(0.0, post_impact_max_ball_vz) * launch_profile["vertical_speed_weight"] * 0.35
        + (760.0 if valid_impact else 0.0)
        - max_offline_y * 900.0
        - backswing_plane_mean * 460.0
        - followthrough_plane_mean * 260.0
        - short_arc * 900.0
        - total_ctrl_energy * 0.000025
    )
    if not hit_ball:
        reward -= 900.0
    if hit_ball and not valid_impact:
        reward -= 320.0

    return {
        "reward": reward,
        "distance": distance,
        "height_gain": height_gain,
        "hit_ball": hit_ball,
        "valid_impact": valid_impact,
        "first_hit_step": first_hit_step,
        "impact_club_vx": impact_club_vx,
        "post_impact_max_ball_vx": post_impact_max_ball_vx,
        "post_impact_max_ball_vz": post_impact_max_ball_vz,
        "post_impact_distance": post_impact_distance,
        "backswing_arc": backswing_arc,
        "on_plane_arc": on_plane_arc,
        "backswing_plane_error": backswing_plane_mean,
        "backswing_plane_score": backswing_plane_score,
        "followthrough_plane_error": followthrough_plane_mean,
        "followthrough_plane_score": followthrough_plane_score,
        "offline_y": max_offline_y,
    }


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
        "arc",
        round(result["backswing_arc"], 4),
        "on_plane_arc",
        round(result["on_plane_arc"], 4),
        "back_plane",
        round(result["backswing_plane_error"], 4),
        "follow_plane",
        round(result["followthrough_plane_error"], 4),
        "offline_y",
        round(result["offline_y"], 4),
        "post_vx",
        round(result["post_impact_max_ball_vx"], 4),
        "post_vz",
        round(result["post_impact_max_ball_vz"], 4),
    )


def train(
    iterations,
    episodes,
    learning_rate,
    std_decay,
    seed=None,
    club_name="7iron",
    hand=DEFAULT_HAND,
):
    rng = random.Random(seed)
    club_name = normalize_club_name(club_name)
    hand = normalize_hand(hand)
    specs = parameter_specs(hand)
    mean = initial_mean(hand)
    std = initial_std()
    model = make_static_model(club_name, hand, club_contact=True, include_actuators=True)
    launch_profile = get_club_launch_profile(club_name)

    best_result = None
    best_candidate = None
    best_valid_result = None
    best_valid_candidate = None
    start_time = time.time()

    print("RL policy trainer:", club_name, hand)
    print("Reward emphasis: backswing arc + backswing plane, with distance/height/launch support")

    for iteration in range(1, iterations + 1):
        returns = []
        epsilons = []
        iteration_best = None
        for _ in range(episodes):
            vector, epsilon = sample_policy_vector(rng, mean, std, specs)
            candidate = vector_to_candidate(vector, hand)
            result = evaluate_rl_swing(model, candidate, launch_profile)
            returns.append(result["reward"])
            epsilons.append(epsilon)

            if iteration_best is None or result["reward"] > iteration_best["reward"]:
                iteration_best = result
            if best_result is None or result["reward"] > best_result["reward"]:
                best_result = result
                best_candidate = candidate
            if result["valid_impact"] and (
                best_valid_result is None or result["reward"] > best_valid_result["reward"]
            ):
                best_valid_result = result
                best_valid_candidate = candidate

        mean, std = update_policy(mean, std, epsilons, returns, specs, learning_rate, std_decay)
        elapsed = time.time() - start_time
        print_result(f"Iteration {iteration}/{iterations}", iteration_best)
        print("valid_found", best_valid_result is not None, "elapsed_sec", round(elapsed, 1))

    final_result = best_valid_result if best_valid_result is not None else best_result
    final_candidate = best_valid_candidate if best_valid_candidate is not None else best_candidate
    print("RL training complete")
    if best_valid_result is None:
        print("No valid RL impact found yet; use the printed candidate only as a shaped-policy checkpoint.")
    print_result("Final best", final_result)
    print_candidate(final_candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="REINFORCE-style policy trainer for the biomech golf arm.")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--std-decay", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--club", choices=sorted(CLUB_PRESETS), default="7iron")
    parser.add_argument("--hand", choices=sorted(HAND_SIGNS), default=DEFAULT_HAND)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        iterations=args.iterations,
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        std_decay=args.std_decay,
        seed=args.seed,
        club_name=args.club,
        hand=args.hand,
    )
