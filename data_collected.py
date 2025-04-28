import numpy as np
import itertools
import random
import torch
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pybullet as p
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm import tqdm

from panda_pushing_env import PandaPushingEnv

class ActionSampler:
    def __init__(self,
                 location_bins=5,
                 angle_bins=7,
                 length_bins=3,
                 location_range=(-1.0, 1.0),
                 angle_range=(-np.pi/2, np.pi/2),
                 length_range=(0.3, 1.0)):
        self.location_vals = np.linspace(*location_range, location_bins)
        self.angle_vals = np.linspace(*angle_range, angle_bins)
        self.length_vals = np.linspace(*length_range, length_bins)

        self.action_list = list(itertools.product(
            self.location_vals,
            self.angle_vals,
            self.length_vals
        ))

    def sample(self):
        return np.array(random.choice(self.action_list), dtype=np.float32)

    def all_actions(self):
        return [np.array(a, dtype=np.float32) for a in self.action_list]

visualizer = None
env = PandaPushingEnv(visualizer=visualizer)
env.reset()
sampler = ActionSampler(location_bins=5, angle_bins=7, length_bins=3)

data = {}
pbar =  tqdm(range(10))
# Perform a sequence of 105 random actions:
for i in pbar:
    state = env.get_state()
    action = sampler.sample()
    print(f"\n action {i}: {action}")
    next_state, reward, done, info = env.step(action)
    data[i] = {
        "state": state,
        "action": action,
        "next_state": next_state,
    }
    if done:
        break

