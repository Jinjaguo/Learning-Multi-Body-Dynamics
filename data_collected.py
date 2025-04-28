import numpy as np
import itertools
import random
import pybullet as p
import os
from multiprocessing import Pool,freeze_support
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


def collect_one_episode(args):
    """
    Collect transitions for a single episode.
    Returns four lists: states, actions, physics_next, next_states.
    """
    ep_id, steps_per_ep, moving_thresh, seed = args
    # seed RNG for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    sampler     = ActionSampler()
    env         = PandaPushingEnv(randomize=True,  debug=False)
    shadow_env  = PandaPushingEnv(randomize=False, debug=False)

    local_states, local_actions, local_physics_next, local_next_states = [], [], [], []

    _ = env.reset()
    full_state = env.get_state()

    for t in range(steps_per_ep):
        action = sampler.sample()

        # physics prediction from shadow env
        fp_next = get_physics_prediction(shadow_env, full_state, action)

        # real environment step
        _ = env.step(action)
        full_state_next = env.get_state()

        # filter out moves where small disk barely moved
        small_cur  = full_state[8:10]
        small_next = full_state_next[8:10]
        if np.linalg.norm(small_next - small_cur) < moving_thresh:
            full_state = full_state_next
            continue

        # cache
        local_states.append(full_state.astype(np.float32))
        local_actions.append(action.astype(np.float32))
        local_physics_next.append(fp_next.astype(np.float32))
        local_next_states.append(full_state_next.astype(np.float32))

        full_state = full_state_next

    return local_states, local_actions, local_physics_next, local_next_states


def main(n_episodes=5000, steps_per_ep=10, moving_thresh=0.002, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4
    os.makedirs("dataset", exist_ok=True)

    args = [
        (ep, steps_per_ep, moving_thresh, ep)
        for ep in range(n_episodes)
    ]

    save_every = 1000  # 满 1000 样本立即保存
    file_idx = 0

    all_states, all_actions, all_phys, all_next = [], [], [], []
    with Pool(n_workers) as pool:
        for states, actions, phys, nexts in tqdm(
                pool.imap_unordered(collect_one_episode, args),
                total=n_episodes,
                desc="Collecting episodes"):
            all_states      .extend(states)
            all_actions     .extend(actions)
            all_phys        .extend(phys)
            all_next        .extend(nexts)
            print(len(all_states))

            while len(all_states) >= save_every:
                chunk = slice(0, save_every)
                np.savez_compressed(f"dataset/part_{file_idx}.npz",
                                    state=np.stack(all_states[chunk]),
                                    action=np.stack(all_actions[chunk]),
                                    physics_next=np.stack(all_phys[chunk]),
                                    next_state=np.stack(all_next[chunk]))
                print(f"[DONE] Collected {len(all_states)} samples")
                # 从列表里弹出已保存的数据，释放内存
                del all_states[:save_every]
                del all_actions[:save_every]
                del all_phys[:save_every]
                del all_next[:save_every]
                file_idx += 1


if __name__ == "__main__":
    freeze_support()  # Windows multiprocessing 必需
    main(
        n_episodes   = 5000,
        steps_per_ep = 10,
        moving_thresh= 0.002,
        n_workers    = 8
    )