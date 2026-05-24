# Best Model Snapshot

This folder restores the `current/` code from commit:

```text
4a83d7a3878535b9305f7fa8d5cbc7d999264535
```

That version produced the best swing results before the project became more
experimental. It is kept separate from the newer `current/` folder so it can be
run, compared, or copied from without mixing it with the newer residual-RL work.

## View The Best Swing

From this folder:

```bash
mjpython human_right_arm_biomech_swing.py --club 7iron --hand right
```

## Train The Best CEM Version

```bash
../.venv/bin/python human_right_arm_biomech_cem.py --club 7iron --hand right
```

## Notes

- This snapshot intentionally uses the older flat file layout from that commit.
- The newer reorganized code remains in `../current/`.
- If the project gets messy again, this folder is the known-good reference point.
