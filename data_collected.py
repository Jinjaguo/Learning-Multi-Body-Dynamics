import numpy as np
import itertools
import random
import pybullet as p
import os
from tqdm import trange

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

def set_env_state(env, full_state):
    """
    Reset the simulation state of both disks to the given full_state.
    full_state: 16-dim array [p_mid(3), v_mid(3), mass_big(1), r_mid(1),
                             p_obj(3), v_obj(3), mass_small(1), r_obj(1)]
    """
    # Parse state
    p_mid      = full_state[0:3]   # [x, y, theta]
    v_mid      = full_state[3:6]   # [vx, vy, omega]
    p_obj      = full_state[8:11]  # [x, y, theta]
    v_obj      = full_state[11:14] # [vx, vy, omega]

    # Convert planar pose to world pose (x,y,z, quaternion)
    mid_world = env._planar_pose_to_world_pose(p_mid)
    obj_world = env._planar_pose_to_world_pose(p_obj)

    pos_mid, orn_mid = mid_world[:3], mid_world[3:]
    pos_obj, orn_obj = obj_world[:3], obj_world[3:]

    # Reset base position & orientation
    p.resetBasePositionAndOrientation(env.mid_objectUid, pos_mid, orn_mid)
    p.resetBasePositionAndOrientation(env.objectUid, pos_obj, orn_obj)

    # Reset velocities: linear [vx, vy, 0], angular [0, 0, omega]
    p.resetBaseVelocity(env.mid_objectUid,
                        linearVelocity=[v_mid[0], v_mid[1], 0.0],
                        angularVelocity=[0.0, 0.0, v_mid[2]])
    p.resetBaseVelocity(env.objectUid,
                        linearVelocity=[v_obj[0], v_obj[1], 0.0],
                        angularVelocity=[0.0, 0.0, v_obj[2]])


def get_physics_prediction(shadow_env, state, action):
    # 1) Reset but保留 sim 句柄（不重新加载 URDF，只重置 state）
    shadow_env.reset()
    set_env_state(shadow_env, state)

    # 2) 在 shadow_env 里执行 action
    _, _, _, _ = shadow_env.step(action)

    # 3) next_state（16维）
    next_state = shadow_env.get_state()

    return next_state



# ====== 超参数 ======
N_EPISODES       = 10        # 采集多少条 episode
STEPS_PER_EP     = 15         # 每个 episode 随机推几步
MOVE_THRESH      = 0.002      # 2mm: 小 disk 位移阈值
OUT_DIR          = "dataset"
OUT_FILE         = "sain_push_dataset.npz"

# ====== 动作采样器 ======
sampler = ActionSampler(location_bins=5, angle_bins=7, length_bins=3)

# ====== 创建环境 ======
env         = PandaPushingEnv(randomize=True,  debug=False)
shadow_env  = PandaPushingEnv(randomize=False, debug=False)

# ====== 数据缓冲区 ======
states, actions, physics_next, next_states = [], [], [], []

for ep in trange(N_EPISODES, desc="Episodes"):
    obs = env.reset()
    full_state = env.get_state()

    for t in range(STEPS_PER_EP):
        action = sampler.sample()

        # --------- physics 预测 ----------
        fp_next = get_physics_prediction(shadow_env, full_state, action)

        # --------- 真环境推进 ----------
        _ , _, _, _ = env.step(action)
        full_state_next = env.get_state()

        # --------- 剔除无效样本 ----------
        small_pose_cur  = full_state[8:11]     # [x,y,θ]   小 disk
        small_pose_next = full_state_next[8:11]

        moved_dist = np.linalg.norm(small_pose_next[:2] - small_pose_cur[:2])
        if moved_dist < MOVE_THRESH:
            # 小盘几乎没动 → 丢弃
            full_state = full_state_next
            continue

        # --------- 缓存 ---------
        states.append(full_state.astype(np.float32))
        actions.append(action.astype(np.float32))
        physics_next.append(fp_next.astype(np.float32))
        next_states.append(full_state_next.astype(np.float32))

        # 准备下一步
        full_state = full_state_next

# ====== 保存 ======
os.makedirs(OUT_DIR, exist_ok=True)
np.savez_compressed(
    os.path.join(OUT_DIR, OUT_FILE),
    state=np.stack(states),
    action=np.stack(actions),
    physics_next=np.stack(physics_next),
    next_state=np.stack(next_states),
)
print(f"[DONE] finished. Num of samples：{len(states)} valid data → {OUT_DIR}/{OUT_FILE}")