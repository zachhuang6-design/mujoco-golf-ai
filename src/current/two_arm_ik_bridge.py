"""Solve or view the IK bridge from kinematic swing to joint model."""

import runpy


if __name__ == "__main__":
    runpy.run_module("golf_core.two_arm_ik_bridge", run_name="__main__")
