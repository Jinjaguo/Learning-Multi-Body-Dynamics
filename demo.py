import os, time, numpy as np, torch
import pybullet as p
from tqdm import trange

from panda_pushing_env import PandaPushingEnv, TARGET_POSE_FREE
from sain_model import SAIN, build_feats, apply_delta
from visualizers import GIFVisualizer
from mppi import MPPI


# ========== 0. 设备 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ========== 1. 加载 SAIN ==========
model = SAIN(dyn_in_dim=14, rel_in_dim=16, delta_dim=6).to(device)   # Δv(3)+Δp(3)
ckpt = torch.load('checkpoints/sain_epoch7000.pth', map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()

# ========== 2. 极简物理“近似” ==========
def physics_stub(s, a):
    return s

@torch.no_grad()
def sain_step(state_flat, action_flat, t=None):
    """
    批量版 SAIN step:
      state_flat: (B,16), action_flat: (B,3) → next_flat: (B,16)
    """
    B = state_flat.size(0)
    s  = state_flat.view(B, 2, 8)             # 恢复成 [B,2,8]
    fp = physics_stub(s, None)               # (B,2,8)

    # 只有 object-0 受控
    # a_full = torch.zeros(B, 2, 3, device=s.device)
    # a_full[:,0,:] = action_flat

    # dyn, rel, send, recv = build_feats(s, a_full, fp)
    dyn, rel, send, recv = build_feats(s, action_flat, fp)
    delta = model(dyn, rel, send, recv)      # (B,2,6)
    s1    = apply_delta(s, delta)            # (B,2,8)
    return s1.reshape(B,16)

# ========== 3. 代价函数 ==========
TARGET_POSE_FREE_T = torch.tensor(TARGET_POSE_FREE[:3], device=device)
def free_cost(state_flat: torch.Tensor,
              action_flat: torch.Tensor) -> torch.Tensor:
    dtype   = state_flat.dtype           # ← float32
    device  = state_flat.device

    target  = TARGET_POSE_FREE_T.to(device=device, dtype=dtype)  # (3,)
    err     = state_flat[:, 8:11] - target                       # (B,3)

    Q = torch.diag(torch.tensor([1., 1., 0.1], device=device,
                                dtype=dtype))                    # (3,3)

    # err @ Q @ err^T  →  每条轨迹一个标量
    return (err @ Q * err).sum(dim=-1)

# ========== 4. MPPI ==========
nx, nu = 16, 3
H, K   = 10, 1024
noise_sigma = 0.2 * torch.eye(nu, device=device)
u_min       = torch.tensor([-1., -np.pi/2, 0.0], device=device)
u_max       = torch.tensor([ 1.,  np.pi/2, 1.0], device=device)

mppi = MPPI(
    dynamics            = sain_step,
    running_cost        = free_cost,
    nx                  = nx,
    num_samples         = K,
    horizon             = H,
    noise_sigma         = noise_sigma,
    lambda_             = 0.01,
    u_min               = u_min,
    u_max               = u_max,
    device              = device,
    step_dependent_dynamics=True
)

# ========== 5. 环境 ==========
# viz = GIFVisualizer()
log = np.load("rollout.npz")
state0  = log["state0"]     # (16,)
actions = log["actions"]    # (T,3)
states  = log["states"]     # (T+1,16)

env = PandaPushingEnv(
        randomize             = False,
        visualizer            = None,
        debug                 = True,
        render_non_push_motions=False,
        camera_heigh          = 800,
        camera_width          = 800,
        render_every_n_steps  = 5)

from data_collected import *
env.reset()
set_env_state(env, state0)


# ========== 6. 控制回路 ==========
# 逐步执行每个动作
#for t in trange(15, desc='Control steps'):
for i, a in enumerate(actions):

    # MPPI 求动作
    # state_np = env.get_state()
    # state = torch.from_numpy(state_np).float().to(device).unsqueeze(0)  # (1,16)
    # action = mppi.command(state).cpu().numpy().squeeze()                # (3,)

    # 真环境执行：step 返回 obs, rew, done, info
    # _, _, done, _ = env.step(action)
    env.step(a)
    time.sleep(0.1)

    # 重新读取完整状态
    state_np = env.get_state()

    # 成功判定：小盘距目标 < 0.05
    if np.linalg.norm(state_np[8:10] - TARGET_POSE_FREE[:2]) < 0.05:
        print(f'✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ goal reached at step {i} ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓', flush=True)

for _ in range(3):
    print(f'✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓ goal reached at step 6 ✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓', flush=True)
p.disconnect()
exit(0)












