# MuJoCo Golf Swing AI

This project explores reinforcement learning under data scarcity. Detailed golf
swing datasets are difficult to access, so I use simulation to create training
environments where AI models can learn, test, and improve swing decisions with
limited real-world data. The same challenge appears in financial modeling,
sports business, and policy-making, where data is often private, incomplete, or
noisy, but decision-makers still need models that can reason through uncertainty
and search for better outcomes.

This project is a physics-based golf swing simulator. It uses MuJoCo to model a
simple golfer arm, wrist, golf club, tee, and ball, then tests how different
joint motions change the swing and the strike.

![Golf Swing Simulation](golf_swing_simulation.png,50%)

The long-term goal is to build toward a more realistic golf swing model. The
project currently focuses on a two-arm + chest + spine swing that can be replayed, measured, and
improved through optimization or AI training. Over time, this can grow into a
more complete body model with two arms, torso rotation, hips, legs, and more
realistic club behavior.

At a high level, the project explores three questions:

- Can a simulated arm move a golf club in a believable swing pattern?
- Can the club make strong, clean contact with the ball?
- Can training methods improve speed, direction, contact quality, and swing
  shape over time?

## Project Structure

- `best_model/`: restored snapshot of the commit that produced the best swing
  results so far.
- `current/`: the active working version of the simulator.
- `current/golf_core/`: the current biomechanical model and working swing
  controller.
- `current/golf_rl/`: AI training and evaluation tools.
- `current/legacy/`: older experiments kept for reference.
- `current/artifacts/`: saved training outputs and model files.

## Current Starting Point

To view the current working swing:

```bash
cd current
mjpython current_swing.py --club 7iron --hand right
```

To view the known-best historical swing:

```bash
cd best_model
mjpython human_right_arm_biomech_swing.py --club 7iron --hand right
```

For the most up-to-date commands and project notes, see:

```bash
current/CURRENT.md
```
