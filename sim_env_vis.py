import torch
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pybullet as p
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm



from panda_pushing_env import PandaPushingEnv
from visualizers import GIFVisualizer



# Initialize the simulation environment. This will only render push motions, omitting the robot reseting motions.
env = PandaPushingEnv(render_non_push_motions=False, camera_heigh=500, camera_width=500, render_every_n_steps=5, debug=True, include_obstacle=True)
env.reset()

# Perform a sequence of 3 random actions:
for i in tqdm(range(50)):
    action_i = env.action_space.sample()
    state, reward, done, info = env.step(action_i)
    if done:
        break


print("Simulation done. Viewer active. Close window or press Ctrl+C to quit.")
while p.isConnected():
    time.sleep(1)

p.disconnect()
