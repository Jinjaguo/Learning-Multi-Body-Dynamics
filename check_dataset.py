import time
import numpy as np
from data_collected import set_env_state
from panda_pushing_env import PandaPushingEnv
import pybullet as p

# 1. 加载数据
data = np.load("dataset/sain_push_dataset.npz")
states         = data["state"]           # (N,16)
actions        = data["action"]          # (N, 3)
next_states    = data["next_state"]      # (N,16)

# 2. 选一条“轨迹”——这里简单取前 T 步
T = 20
traj_states  = states[:T+1]   # 0...T
traj_actions = actions[:T]    # 0...T-1

# 3. 启动 env（GUI 模式）并重置
env = PandaPushingEnv(
    randomize=False,      # playback 用固定参数
    debug=True,           # GUI
    render_non_push_motions=True
)
env.reset()

# 4. 手动把 env 同步到轨迹初始状态
set_env_state(env, traj_states[0])

# 5. 回放每一步动作
for t in range(T):
    a = traj_actions[t]
    env.step(a)
    time.sleep(0.5)

# 6. 挂起，直到你手动关闭 GUI
while p.isConnected():
    time.sleep(1)
