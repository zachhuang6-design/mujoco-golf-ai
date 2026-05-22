import argparse
import math
import random
import time

import mujoco

import golf_3joint_common
from golf_3joint_common import make_single_arm_model, print_candidate, print_copy_paste_controls
from human_random_ai_trained3joint import simulate_swing


PARAMS = [
    ("t1", 80.0, 280.0),
    ("t2_gap", 35.0, 180.0),
    ("t3_gap", 35.0, 180.0),
    ("s1", 0.0, 10.0),
    ("e1", 0.0, 4.0),
    ("w1", 0.0, 4.0),
    ("s2", -10.0, -2.0),
    ("e2", -4.0, 1.5),
    ("w2", -4.0, 1.0),
    ("s3", -10.0, 0.0),
    ("e3", -4.0, 0.0),
    ("w3", -4.0, -0.8),
]

INITIAL_MEAN = [
    180.0,
    85.0,
    90.0,
    6.5,
    2.0,
    3.0,
    -8.0,
    -0.8,
    -0.6,
    -7.0,
    -3.0,
    -3.4,
]

INITIAL_STD = [
    45.0,
    35.0,
    35.0,
    2.5,
    1.2,
    1.0,
    1.8,
    1.2,
    1.0,
    2.0,
    0.9,
    0.5,
]

MIN_STD = [
    6.0,
    6.0,
    6.0,
    0.25,
    0.15,
    0.15,
    0.25,
    0.15,
    0.15,
    0.25,
    0.15,
    0.15,
]


def clip(value, low, high):
    return max(low, min(high, value))


def sample_vector(rng, mean, std):
    vector = []
    for i, (_, low, high) in enumerate(PARAMS):
        value = rng.gauss(mean[i], std[i])
        vector.append(clip(value, low, high))
    return vector


def vector_to_candidate(vector):
    t1 = int(round(vector[0]))
    t2 = int(round(t1 + vector[1]))
    t3 = int(round(t2 + vector[2]))
    return {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "s1": vector[3],
        "e1": vector[4],
        "w1": vector[5],
        "s2": vector[6],
        "e2": vector[7],
        "w2": vector[8],
        "s3": vector[9],
        "e3": vector[10],
        "w3": vector[11],
    }


def mean(values):
    return sum(values) / len(values)


def stddev(values, center):
    if len(values) < 2:
        return 0.0
    variance = sum((value - center) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


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
    active_min_tip = result.get("active_min_tip_to_ball", result.get("min_tip_to_ball", math.inf))
    active_max_club_vx = result.get("active_max_club_vx", 0.0)
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
        "active_min_tip",
        round(active_min_tip, 4),
        "active_max_vx",
        round(active_max_club_vx, 4),
    )


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
    print("Initial ball position:", [round(v, 4) for v in ball_pos])
    print("Initial club tip position:", [round(v, 4) for v in tip_pos])
    print("Initial tip-to-ball distance:", round(math.dist(tip_pos, ball_pos), 4))
    print("Initial contacts:", data.ncon)


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
            result = simulate_swing(model, candidate, require_hit=True)
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
        elite_vectors = [item[1] for item in scored[:elite_count]]
        mean_vector, std_vector = update_distribution(elite_vectors, mean_vector, std_vector, smoothing)

        generation_best = scored[0][3]
        elapsed = time.time() - start_time
        print_result(f"Generation {generation}/{generations}", generation_best)
        print(
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
    print_result("Final best", final_result)
    print("Impact club vx:", round(final_result["impact_club_vx"], 4))
    print("Impact ball vx:", round(final_result["impact_ball_vx"], 4))
    print("Impact ball vz:", round(final_result["impact_ball_vz"], 4))
    print("Post-impact max ball vx:", round(final_result.get("post_impact_max_ball_vx", 0.0), 4))
    print("Post-impact max ball vz:", round(final_result.get("post_impact_max_ball_vz", 0.0), 4))
    print("Post-impact distance:", round(final_result.get("post_impact_distance", 0.0), 4))
    print("Sequence score:", round(final_result["sequence_score"], 4))
    print("Minimum tip-to-ball distance:", round(final_result["min_tip_to_ball"], 4))
    print("Active downswing minimum tip-to-ball distance:", round(final_result.get("active_min_tip_to_ball", final_result["min_tip_to_ball"]), 4))
    print("Active downswing max club vx:", round(final_result.get("active_max_club_vx", 0.0), 4))
    print("Active downswing max club vz:", round(final_result.get("active_max_club_vz", 0.0), 4))
    print_candidate(final_candidate)
    if final_result["valid_impact"]:
        print_copy_paste_controls(final_candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Entropy Method trainer for 3-joint golf swing controls.")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elite-count", type=int, default=12)
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