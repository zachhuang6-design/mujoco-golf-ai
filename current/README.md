# MuJoCo Golf AI

Start with `CURRENT.md`.

The active code is intentionally small:

- `current_swing.py` launches the current working swing viewer.
- `golf_core/` contains the current biomechanical model and CEM/PD swing.
- `golf_rl/` contains the residual-RL training/evaluation tools.
- `legacy/` contains older scripts that are preserved but not part of the current path.
- `artifacts/` contains generated runs and trained models.

Recommended first check:

```bash
mjpython current_swing.py --club 7iron --hand right
```
