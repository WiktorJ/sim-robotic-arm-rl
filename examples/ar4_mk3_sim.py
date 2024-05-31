import time
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("/Users/wiktorjurasz/repos/Gymnasium-Robotics/gymnasium_robotics/envs/assets/fetch/reach.xml")
model = mujoco.MjModel.from_xml_path("/Users/wiktorjurasz/Projects/sim-robotic-arm-rl/mujoco/ar4_mk3/scene.xml")
data = mujoco.MjData(model)
n_steps = 5

# viewer shows frame of environment every n_steps
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while True:
        step_start = time.time()
        for _ in range(n_steps):
            mujoco.mj_step(model, data)
        viewer.sync()
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)