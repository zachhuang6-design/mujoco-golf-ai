"""Run the joint-accurate two-arm physical baseline swing."""

import runpy


if __name__ == "__main__":
    runpy.run_module("golf_core.two_arm_joint_physics", run_name="__main__")
