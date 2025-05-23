from stable_baselines3 import SAC
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import sem, t
from scipy.signal import welch
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FFMpegWriter

# CONFIG
log_root_path = os.path.abspath(os.path.join("logs", "sb3", "Project"))
times = "2025-05-21_13-51-38"
model_names = [f"SAC_{times}_seed{i}" for i in range(3)]

n_episodes = 20
fixed_seed = 123
success_threshold = 100
obs_indices = [4, 6, 9, 11]
title = ["HIP1", "KNEE1", "HIP2", "KNEE2"]

# ENV
env = DummyVecEnv([lambda: gym.make("BipedalWalker-v3", render_mode=None)])

# เก็บข้อมูลทั้งหมดของทุก episode
episode_data = []

for model_idx, model_name in enumerate(model_names):
    print(f"\n[INFO] Evaluating model: {model_name}")
    model = SAC.load(f"{log_root_path}/{model_name}/model", env=env)

    for ep in range(n_episodes):
        seed = fixed_seed + ep
        env.envs[0].reset(seed=seed)
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        obs_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            step_count += 1
            obs_history.append(obs[0])

        obs_history = np.array(obs_history)
        selected_obs = obs_history[:, obs_indices]

        from scipy.stats import gmean

        def compute_spectral_flatness(signal, sf=1.0):
            freqs, psd = welch(signal, sf, nperseg=min(256, len(signal)))
            flatness = gmean(psd + 1e-12) / (np.mean(psd) + 1e-12)
            return flatness

        episode_flatness = np.mean([compute_spectral_flatness(selected_obs[:, i]) for i in range(4)])

        episode_data.append({
            "model_name": model_name,
            "model_idx": model_idx,
            "episode": ep,
            "seed": seed,
            "reward": total_reward,
            "steps": step_count,
            "success": total_reward >= success_threshold,
            "obs": selected_obs,
            "entropy": episode_flatness
        })

# ===== วิเคราะห์รวม =====
def ci95(x):
    return sem(x) * t.ppf((1 + 0.95) / 2, len(x) - 1)

all_rewards = np.array([e["reward"] for e in episode_data])
all_steps = np.array([e["steps"] for e in episode_data])
all_success = np.array([e["success"] for e in episode_data])
all_entropy = np.array([e["entropy"] for e in episode_data])

print("\n========== FINAL COMBINED SUMMARY ==========")
print(f"✅ Total Episodes         : {len(episode_data)}")
print(f"🎯 Reward Mean ± CI95     : {np.mean(all_rewards):.2f} ± {ci95(all_rewards):.2f}")
print(f"⏱  Steps Mean ± CI95      : {np.mean(all_steps):.2f} ± {ci95(all_steps):.2f}")
print(f"✅ Success Rate ± CI95     : {np.mean(all_success)*100:.2f}% ± {ci95(all_success)*100:.2f}%")
print(f"📈 Spectral Entropy ± CI95 : {np.mean(all_entropy):.4f} ± {ci95(all_entropy):.4f}")

# ===== ดึง episode ที่ได้ reward สูงที่สุดจริง ๆ =====
best_ep = max(episode_data, key=lambda e: e["reward"])

print(f"\n🏆 Best Episode: {best_ep['model_name']} | Episode {best_ep['episode']} | Reward = {best_ep['reward']:.2f}")
print(f"📉 Flatness (Best Episode): {best_ep['entropy']:.4f}")

# ===== เซฟกราฟ OBS =====
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(best_ep["obs"][:, i])
    plt.title(title[i])
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)
plt.suptitle(f"SAC - Observation Time-Series (Best Episode Reward = {best_ep['reward']:.2f})")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("SAC_best_episode_observation.png")
plt.close()

# ===== เซฟกราฟ FFT =====
plt.figure(figsize=(10, 6))
for i in range(4):
    signal = best_ep["obs"][:, i]
    freqs, psd = welch(signal, nperseg=min(256, len(signal)))
    plt.subplot(2, 2, i + 1)
    plt.semilogy(freqs, psd)
    plt.title(f"FFT of {title[i]}")
    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.grid(True)
plt.suptitle("SAC - FFT Spectrum of Observation (Best Episode)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("SAC_best_episode_fft.png")
plt.close()

# ===== บันทึกวิดีโอจริงจากการ render =====
render_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
model = SAC.load(f"{log_root_path}/{best_ep['model_name']}/model", env=render_env)
obs, _ = render_env.reset(seed=best_ep['seed'])
done = False
frames = []

while not done:
    frame = render_env.render()
    frames.append(frame)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = render_env.step(action)

fig = plt.figure(figsize=(6, 6))
plt.axis("off")
im = plt.imshow(frames[0])

def update(i):
    im.set_array(frames[i])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=30, blit=True)
writer = FFMpegWriter(fps=30)
ani.save("SAC_best_episode_video.mp4", writer=writer)
plt.close(fig)

print("🎥 Saved episode video to: SAC_best_episode_video.mp4")
