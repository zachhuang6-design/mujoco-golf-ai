import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

print("Ball position:", data.qpos[:3])
