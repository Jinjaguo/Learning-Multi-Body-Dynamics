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
N = len(actions)

# 2. 设定回放长度
T = 19 # 每次回放 T 步

start = np.random.randint(0, N - T + 1)
end = start + T

traj_states  = states[start : end+1]   # state0 ... stateT （共 T+1 条）
traj_actions = actions[start : end]    # action0 ... action(T-1)（共 T 条）

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

# 6. 挂起，直到你手动关闭 GUI
while p.isConnected():
    time.sleep(1)
