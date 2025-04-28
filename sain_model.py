# sain_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
from data_processing import process_data_single_step
DT = 1.0 / 240.0


# ---------- 1. 四层 MLP ---------------
class FourLayerMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128,  64),    nn.ReLU(),
            nn.Linear(64,   32),    nn.ReLU(),
            nn.Linear(32,   16),    nn.ReLU(),
            nn.Linear(16, out_dim)
        )
    def forward(self, x): return self.net(x)


# ---------- 2.  Interaction Network (IN) ----------
class InteractionNetwork(nn.Module):
    """
    纯数据驱动 IN：输入一批对象和 pairwise 边，输出 Δstate。
    对每个 object:
        dyn_input  = [v_i, a_i, m_i, r_i]          (可自行扩展)
        rel_input  = [v_i,  p_i - p_j, v_i - v_j,
                      m_i,m_j,r_i,r_j]             (与论文一致)
    """
    def __init__(self,
                 dyn_in_dim=  4+2,    # vx,vy,ω + a, plus other terms
                 rel_in_dim=  8+4,    # 见上面注释
                 delta_dim  =  3+3):  # Δv (3) + 预留 Δp (3)
        super().__init__()
        self.frel = FourLayerMLP(rel_in_dim, 16)
        self.fdyn = FourLayerMLP(dyn_in_dim + 16, delta_dim)

    def forward(self,
                dyn_feats,               # [B,N,d_dyn]
                rel_feats,               # [B,E,d_rel]
                senders, receivers):     # shape [E]  long tensor
        """
        senders/receivers: 边索引，值域 [0,N-1]
        """
        B, N, _ = dyn_feats.shape
        E = rel_feats.size(1)

        # --- frel: message on each edge ---
        rel_in  = rel_feats.view(-1, rel_feats.size(-1))   # [B*E, rel_in_dim]
        msg     = self.frel(rel_in).view(B, E, 16)         # [B,E,16]

        # --- 聚合消息到每个 object（加法） ---
        agg = torch.zeros(B, N, 16, device=dyn_feats.device)
        agg.index_add_(1, receivers, msg)   # 按 receiver 聚合

        # --- fdyn: 预测 Δv ---
        dyn_in = torch.cat([dyn_feats, agg], dim=-1)       # [B,N,d_dyn+16]
        delta  = self.fdyn(dyn_in)                         # [B,N,delta_dim]
        return delta


# ---------- 3.  Simulator-Augmented IN (SAIN) ----------
class SAIN(nn.Module):
    def __init__(self, dyn_in_dim=14, rel_in_dim=16, delta_dim=6):
        super().__init__()
        self.frel = FourLayerMLP(rel_in_dim, 16)
        self.fdyn = FourLayerMLP(dyn_in_dim + 16, delta_dim)  # 多拼 16D 消息

    def forward(self, dyn, rel, send, recv):
        B,N,_ = dyn.shape; E = rel.size(1)
        msg = self.frel(rel.view(-1,rel.size(-1))).view(B,E,16)
        agg = torch.zeros(B,N,16, device=dyn.device)
        agg.index_add_(1, recv, msg)
        return self.fdyn(torch.cat([dyn, agg], -1))       # [B,N,6]


def hybrid_loss(pred, gt):
    p_pred, v_pred, θ_pred = pred[...,:2], pred[...,3:5], pred[...,2]
    p_gt,   v_gt,   θ_gt   = gt  [...,:2], gt  [...,3:5], gt  [...,2]
    return F.mse_loss(p_pred,p_gt)+F.mse_loss(v_pred,v_gt)+ \
           F.mse_loss(torch.sin(θ_pred),torch.sin(θ_gt))+ \
           F.mse_loss(torch.cos(θ_pred),torch.cos(θ_gt))

def build_feats(state, action, phys_pred):
    """
    state, phys_pred : [B,2,8]
    action           : [B,3]  (loc,angle,len)
    返回 dyn_feats [B,2,14] , rel_feats[B,2,16], senders/receivers [2]
    """
    B = state.size(0)
    p  = state[...,:3]
    v  = state[...,3:6]
    m  = state[...,6:7]
    r  = state[...,7:]

    phys_delta = phys_pred - state
    pd_pos = phys_delta[...,:3]
    pd_vel = phys_delta[...,3:6]

    # ---- dyn_feats ----
    a_tile = torch.zeros(B,2,3, device=state.device)
    a_tile[:,0,:] = action                      # 只有物体0 有 push
    dyn_feats = torch.cat([v, a_tile, m, r, pd_pos, pd_vel], dim=-1)  # 3+3+1+1+3+3 = 14

    # ---- rel_feats ----
    senders    = torch.tensor([0,1], dtype=torch.long, device=state.device)
    receivers  = torch.tensor([1,0], dtype=torch.long, device=state.device)
    p_i, p_j   = p[:,senders], p[:,receivers]
    v_i, v_j   = v[:,senders], v[:,receivers]
    rel_feats  = torch.cat([v_i,
                            p_i - p_j,
                            v_i - v_j,
                            m[:,senders], m[:,receivers],
                            r[:,senders], r[:,receivers],
                            pd_vel[:,senders]], dim=-1)        # 3+3+3+1+1+1+1+3=16
    return dyn_feats, rel_feats, senders, receivers


def apply_delta(state, delta):                                       # ②
    dv, dp = delta[...,:3], delta[...,3:6]        # 6-dim split
    v_new  = state[...,3:6] + dv
    p_new  = state[...,:3]  + dp                  # 直接加 Δp
    return torch.cat([p_new, v_new, state[...,6:]], -1)


def hybrid_loss(pred, gt):                                           # ③
    p_pred, v_pred, th_pred = pred[...,:2], pred[...,3:5], pred[...,2]
    p_gt,   v_gt,   th_gt   = gt  [...,:2], gt  [...,3:5], gt  [...,2]
    return (F.mse_loss(p_pred,p_gt) + F.mse_loss(v_pred,v_gt) +
            F.mse_loss(torch.sin(th_pred),torch.sin(th_gt)) +
            F.mse_loss(torch.cos(th_pred),torch.cos(th_gt)))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, val_loader = process_data_single_step(
        "dataset",  # 合并后的 npz
        batch_size=16)

    model = SAIN().to(device)
    opt   = torch.optim.Adam(model.parameters(), 1e-3)
    sch   = torch.optim.lr_scheduler.StepLR(opt, 2500, 0.5)

    best_val = 1e9; save_every = 500; epochs = 8000
    for epoch in trange(1, epochs+1, desc="Epochs"):
        # ---- Train ----
        model.train(); loss_sum = 0.
        for batch in train_loader:
            s   = batch['state']       .view(-1,2,8).to(device)
            fp  = batch['physics_next'].view(-1,2,8).to(device)
            sgt = batch['next_state']  .view(-1,2,8).to(device)
            a   = batch['action']                      .to(device)

            dyn, rel, send, recv = build_feats(s,a,fp)
            delta = model(dyn,rel,send,recv)
            pred  = apply_delta(s,delta)
            loss  = hybrid_loss(pred,sgt)

            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()*s.size(0)
        train_loss = loss_sum/len(train_loader.dataset)

        sch.step()

        # ---- Val & save ----
        if epoch % save_every == 0:
            model.eval(); vloss = 0.
            with torch.no_grad():
                for batch in val_loader:
                    s   = batch['state']      .view(-1,2,8).to(device)
                    fp  = batch['physics_next'].view(-1,2,8).to(device)
                    sgt = batch['next_state'] .view(-1,2,8).to(device)
                    a   = batch['action']                     .to(device)
                    dyn,rel,send,recv = build_feats(s,a,fp)
                    pred = apply_delta(s, model(dyn,rel,send,recv))
                    vloss += hybrid_loss(pred,sgt).item()*s.size(0)
            vloss /= len(val_loader.dataset)

            print(f"\nEpoch {epoch:4d}  Train {train_loss:.6f}  Val {vloss:.6f}")

            torch.save({'epoch':epoch,
                        'model_state':model.state_dict()},
                       f"checkpoints/sain_epoch{epoch}.pth")
            if vloss < best_val:
                best_val = vloss
                torch.save(model.state_dict(), "checkpoints/sain_best.pth")
                print(f"★ New best {best_val:.6f}")