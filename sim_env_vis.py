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
from tqdm import tqdm



from data_collected import *
from panda_pushing_env import PandaPushingEnv
from visualizers import GIFVisualizer



# Initialize the simulation environment. This will only render push motions, omitting the robot reseting motions.
env = PandaPushingEnv(render_non_push_motions=False, camera_heigh=500, camera_width=500, debug=True)
env.reset()
sampler = ActionSampler(location_bins=5, angle_bins=7, length_bins=3)

# Perform a sequence of 105 random actions:
for i in tqdm(range(15)):
    action_i = sampler.sample()
    print(f"\n action {i}: {action_i}")
    state, reward, done, info = env.step(action_i)
    if done:
        break

while p.isConnected():
    time.sleep(1)

p.disconnect()
