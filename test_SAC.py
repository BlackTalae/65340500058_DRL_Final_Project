from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import gymnasium as gym
from datetime import datetime
import os
import gc
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

# ==== CONFIG ====
seeds = [0, 1, 2, 3, 4]

log_root_path = os.path.abspath(os.path.join("logs", "sb3", "Project"))
print(f"[INFO] Logging experiment in directory: {log_root_path}")
run_info = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
log_dirs = []  # เก็บ path log แต่ละ run

# ==== TRAINING LOOP ====
for seed in seeds:
    
    log_dir = os.path.join(log_root_path, f"SAC_{run_info}_seed{seed}")
    print(f"\n[INFO] Running SAC with seed={seed}, log_dir={log_dir}")
    os.makedirs(log_dir, exist_ok=True)
    log_dirs.append(log_dir)

    # 1. Create environment
    env = gym.make("BipedalWalker-v3")

    # 2. Create SAC agent
    agent = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,           # ✅ Default ที่ดีสำหรับ SAC
        buffer_size=100000,         # ✅ ใหญ่พอสำหรับ off-policy
        batch_size=256,              # ✅ ใหญ่กว่าของ PPO ได้
        tau=0.005,                   # ✅ Soft target update (default)
        gamma=0.99,                  # ✅ Discount factor
        train_freq=1,                # ✅ Train every step
        gradient_steps=1,            # ✅ Update 1 step ต่อ train step
        ent_coef="auto_0.1",         # ✅ Entropy coefficient ที่ปรับเองได้
        learning_starts=1000,        # ✅ เริ่มเรียนรู้หลังเก็บ buffer แล้ว
        use_sde=True,                # ✅ State-dependent exploration (ช่วยใน continuous env)
        sde_sample_freq=4,           # ✅ Sample exploration noise ทุก ๆ 4 steps
        seed=seed,
        verbose=1
    )

    # 3. Logger
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    # 4. Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=log_dir,
        name_prefix="model",
        verbose=1
    )

    # 5. Train
    agent.learn(total_timesteps=500000, callback=checkpoint_callback)

    # 6. Save
    agent.save(os.path.join(log_dir, "model"))

    # 7. Cleanup
    del agent, env
    gc.collect()

# ==== LOAD LOGS MULTI-TAG ====

def load_scalar(logdir, tag):
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        print(f"[WARN] Tag '{tag}' not found in {logdir}")
        return None, None
    scalars = ea.Scalars(tag)
    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]
    return np.array(steps), np.array(values)

# ===== TAGS ที่ต้องการเฉลี่ย ====
tags_to_plot = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean"
]

# ===== LOG AVERAGE INTO TENSORBOARD =====
avg_logdir = os.path.join(log_root_path, f"SAC_avg_{run_info}")
writer = SummaryWriter(log_dir=avg_logdir)

print("\n[INFO] Loading logs and writing average to TensorBoard...")

for tag in tags_to_plot:
    all_values = []
    common_steps = None
    min_length = float("inf")

    for d in log_dirs:
        steps, values = load_scalar(d, tag)
        if steps is None or len(values) < 2:
            continue
        if common_steps is None:
            common_steps = steps
        min_length = min(min_length, len(values))
        all_values.append(values)

    if len(all_values) == 0:
        print(f"[WARN] No valid runs found for tag: {tag}")
        continue

    values_aligned = np.array([v[:min_length] for v in all_values])
    common_steps = common_steps[:min_length]
    mean = np.mean(values_aligned, axis=0)

    for step, m in zip(common_steps, mean):
        writer.add_scalar(tag, m, step)

writer.close()
print(f"[INFO] Average values logged to TensorBoard at: {avg_logdir}")

