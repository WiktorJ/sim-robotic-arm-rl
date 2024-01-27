from dm_control import suite
#@title All `dm_control` imports required for this tutorial

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation

# General
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image

#@title Listing all `manipulation` tasks{vertical-output: true}


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 32
    orig_backend = matplotlib.get_backend()
    matplotlib.use(
        'Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    plt.show()
    # anim.save('animated_arm.gif', fps=24, writer='imagemagick')
    anim.save('animated_arm.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
    # return HTML(anim.to_html5_video())

# `ALL` is a tuple containing the names of all of the environments in the suite.
# print('\n'.join(manipulation.ALL))
#
# print('----vision----')
#
# #@title Listing `manipulation` tasks that use vision{vertical-output: true}
# print('\n'.join(manipulation.get_environments_by_tag('vision')))


#@title Loading and simulating a `manipulation` task{vertical-output: true}
from dm_control import suite
env = manipulation.load('stack_2_of_3_bricks_random_order_vision', seed=42)
# env = suite.load(domain_name="cartpole", task_name="swingup")
action_spec = env.action_spec()
observation_spec = env.observation_spec()
print(env.reset())
# print(dir(env))
# print(dir(env.step(action_spec.generate_value())))
# print(env.step(action_spec.generate_value()))
#
# print(dir(action_spec))
# print(dir(observation_spec))
# print(dir(observation_spec))
# print(observation_spec)
# print(action_spec.generate_value())
# print(action_spec.shape)
# print(action_spec.minimum)
# print(action_spec.maximum)
# def sample_random_action():
#   return env.random_state.uniform(
#       low=action_spec.minimum,
#       high=action_spec.maximum,
#   ).astype(action_spec.dtype, copy=False)
#
# # Step the environment through a full episode using random actions and record
# # the camera observations.
# frames = []
# timestep = env.reset()
# frames.append(timestep.observation['front_close'])
# while not timestep.last():
#   timestep = env.step(sample_random_action())
#   frames.append(timestep.observation['front_close'])
# all_frames = np.concatenate(frames, axis=0)
# display_video(all_frames, 30)