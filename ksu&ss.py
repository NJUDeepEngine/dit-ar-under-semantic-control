# 超参数
K_patch = 6           # 帧内反传长度
K_time  = 1           # 往前看的历史帧数量（仅参与前向，梯度在帧边界截断）
p_ss    = schedule_p(epoch)   # 0 -> 0.3
row     = 8                   # 每行8个patch，P=64

# 选一个 t0 = (t_frame0, p0)
t_frame0 = randint(0, T-1)
p0       = randint(0, P-1)    # 或 50% 概率令 p0=0（对齐到帧首）

# 1) 构造“可见前缀”：最近 K_time 个历史帧 + 当前帧的 [0..p0-1]
#    历史帧用 SS 脏化（用 EMA teacher），当前帧的前缀也可 SS
ctx_seq = []
with torch.no_grad():
    # 历史帧（最多 K_time 帧）
    for dt in range(K_time, 0, -1):
        tf = max(t_frame0 - dt, 0)
        # 只取这帧的全64个patch作为上下文（不会对这帧反传梯度）
        hist = seq_gt[:, tf, :, ...]  # (B, P, C, ps, ps)
        hist_ctx = ss_frame(hist, teacher=ema_model, p=p_ss, kwargs=causal_kwargs)
        ctx_seq.append(hist_ctx)      # 参与前向，稍后整体 detach

    # 当前帧的前缀 [0..p0-1]
    cur_prefix = ss_prefix_in_frame(seq_gt[:, t_frame0, :, ...], p0, teacher=ema_model, p=p_ss, kwargs=causal_kwargs)
    ctx_seq.append(cur_prefix)        # (B, p0, C, ps, ps)

ctx = torch.cat(ctx_seq, dim=1)       # (B, K_time*P + p0, C, ps, ps)
ctx = ctx.detach()                    # 历史部分在帧边界截断 

# 2) 窗口内 K_patch 步：在“当前帧”从 p0 开始自喂，逐步预测 p0..p0+K_patch-1
loss = 0.0
weights = [0.8**k for k in range(K_patch)]
for k in range(K_patch):
    p = p0 + k
    if p >= P: break  # 不跨帧（常规做法）；偶尔允许跨到下一帧另开几步也行

    pred = model(ctx, **causal_kwargs)                 # (B,1,C,ps,ps)，预测当前帧的第 p 个 patch
    gt   = seq_gt[:, t_frame0, p:p+1, ...]
    loss += weights[k] * patch_loss(pred, gt)          # latent L1/Huber (+ 可选小LPIPS)

    # 自喂 + 保持窗口内全反传
    ctx  = torch.cat([ctx, pred], dim=1)

loss = loss / sum(weights)
loss.backward()
opt.step(); ema.update(model)
