The current biomechanics model is generated from `human_right_arm_biomech_static.py`
so club presets and handedness stay in one source of truth.

`GolfSwingEnv` can still load a raw MuJoCo XML path through `model_path`, but the
default path is the generated right-arm model with the selected club preset.

