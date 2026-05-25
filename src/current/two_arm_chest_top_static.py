"""Run the static two-arm top-of-backswing viewer."""

import runpy


if __name__ == "__main__":
    runpy.run_module("golf_core.two_arm_chest_top_static", run_name="__main__")
