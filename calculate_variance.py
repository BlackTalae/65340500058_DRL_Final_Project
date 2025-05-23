import pickle
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, SAC
import gymnasium as gym
import torch
import os

with open("SAC_episode_data.pkl", "rb") as f:
    episode_data = pickle.load(f)

print(f"Loaded {len(episode_data)} episodes")


# ===== Generalization Variance: Reward variance across models per env seed =====
from collections import defaultdict

reward_by_seed = defaultdict(list)

for ep in episode_data:
    reward_by_seed[ep["seed"]].append(ep["reward"])

reward_variance_per_seed = []
for seed, rewards in reward_by_seed.items():
    print(rewards)
    if len(rewards) >= 2:
        reward_var = np.var(rewards, ddof=1)  # sample variance
        reward_variance_per_seed.append({
            "seed": seed,
            "reward_var": reward_var
        })
    else:
        print(f"[WARN] Seed {seed} has <2 models. Skipped.")

reward_var_df = pd.DataFrame(reward_variance_per_seed)

print("\n========== Reward Variance across Models per Seed ==========")
print(reward_var_df.head())
print(reward_var_df.shape)
print(f"\nMean reward variance across seeds: {reward_var_df['reward_var'].mean():.4f}")
reward_var_df.to_csv("reward_variance_per_seed.csv", index=False)

# ======================================================================================================

def compute_action_mse(models, env, num_states=100, device="cpu"):
    import torch
    import numpy as np

    states = np.array([env.observation_space.sample() for _ in range(num_states)])
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)

    mean_actions = []
    for model in models:
        with torch.no_grad():
            # PPO: use predict with deterministic=True
            if hasattr(model.policy, "predict"):
                actions, _ = model.policy.predict(states_tensor.cpu().numpy(), deterministic=True)
                actions = torch.tensor(actions, dtype=torch.float32).to(device)
            # SAC/TD3: use actor directly
            elif hasattr(model.policy, "actor"):
                actions = model.policy.actor(states_tensor)
            else:
                raise ValueError("Unsupported policy type for action extraction.")
            mean_actions.append(actions)

    # Compute pairwise MSE
    mse_scores = []
    for i in range(len(mean_actions)):
        for j in range(i + 1, len(mean_actions)):
            diff = mean_actions[i] - mean_actions[j]
            mse = torch.mean(diff ** 2, dim=-1).cpu().numpy()  # [n_states]
            mse_scores.append(mse)

    all_mse = np.concatenate(mse_scores)
    return {
        "mse_mean": all_mse.mean(),
        "mse_std": all_mse.std(),
        "mse_all": all_mse
    }

log_root_path = os.path.abspath(os.path.join("logs", "sb3", "Project"))
times = "2025-05-22_08-54-48"
model_names = [f"SAC_{times}_seed{i}" for i in range(5)]

best_ep = max(episode_data, key=lambda e: e["reward"])
env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode=None)])
models = [SAC.load(os.path.join(log_root_path, name, "model"), env=env) for name in model_names]

result = compute_action_mse(models, env, num_states=100, device="cuda")
print(f"Mean MSE: {result['mse_mean']:.6f} ± {result['mse_std']:.6f}")

