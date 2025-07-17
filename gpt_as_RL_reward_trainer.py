from __future__ import annotations
from tqdm import tqdm

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import SlideToGoalEnv
from gpt_client import GPTClient

system_prompt = """
You are given two rendered images of a game environment: one from a side view and one from a top-down view. The scene contains a blue cube that slides from the origin (0, 0) toward a red circular goal area. The ground is made of 1-meter-by-1-meter tiles for scale reference. The red circular goal has a radius of 0.100 meters.

Your task is to estimate the RL reward after the slide based on the cube’s final position. The reward is defined as the negative Euclidean distance in meters between the center of the blue cube and the center of the red goal area. If the cube appears to be perfectly centered on the goal, return 0.000. Otherwise, return a negative distance value in the format -0.123.

---

**Examples for reference only:**

- If the cube is perfectly centered on the red circle, the reward is: `0.000`.

- If the cube is located at a corner of the top-down view and the red goal is in the center, the diagonal distance is about 0.7 meters. The reward would be approximately: `-0.700`.

- If the cube is located exactly one tile (1 meter) to the right of the goal, the reward is approximately: `-1.000`.

- If the cube is slightly off from the center, for example 0.2 meters away in any direction, the reward would be approximately: `-0.200`.

- If the cube is close to the edge of the red goal (about 0.1 meters away), the reward would be around: `-0.100`.

---

Output only the scalar reward in `-0.123` format. Do not include any explanation or extra text.

"""


def gpt_reward_estimation(env: SlideToGoalEnv, top_img: str, side_img: str, client: GPTClient) -> tuple[float, float]:
    """Estimate reward using GPT and return (gpt_reward, ground_truth_reward)."""
    final_pos = env.get_state()
    goal = env.goal_position
    gt_reward = -np.linalg.norm(final_pos[:2] - goal[:2])

    top_id = client.upload_image(top_img)
    side_id = client.upload_image(side_img)

    messages = [
        {"role": "user", "content": [
            {"type": "input_image", "file_id": top_id},
            {"type": "input_image", "file_id": side_id},
        ]}
    ]

    output = client.chat(system_prompt, messages)
    gpt_reward = float(output)
    if gt_reward < -0.71:
        gpt_reward = -1
    return gpt_reward, gt_reward


def train(iterations: int = 100, batch_size: int = 20, eta: float = 1.0, force_clip: float = 100.0, action_dim: int = 2) -> None:
    os.makedirs("images", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = SlideToGoalEnv(gui=False)
    client = GPTClient()  # Add your own client init logic here

    # GT Agent
    mean_gt = np.zeros(action_dim)
    cov_gt = np.eye(action_dim) * 1000

    # GPT Agent
    mean_gpt = np.zeros(action_dim)
    cov_gpt = np.eye(action_dim) * 1000

    logs = {
        "gt": {"rewards": [], "avg_rewards": [], "cov_trace": [], "mean_hist": []},
        "gpt": {"rewards": [], "avg_rewards": [], "cov_trace": [], "mean_hist": []}
    }

    for iteration in tqdm(range(iterations)):
        # === Sample actions ===
        actions_gt = np.random.multivariate_normal(mean_gt, cov_gt, batch_size)
        #edit this to mean_gt if you can to have same actions for both agents
        actions_gpt = np.random.multivariate_normal(mean_gpt, cov_gpt, batch_size)

        gt_rewards, gpt_rewards = [], []

        # === Evaluate GT agent ===
        for i, a in enumerate(actions_gt):
            a = np.clip(a, -force_clip, force_clip)
            force = [a[0], a[1], 0.0]

            env.reset()
            env.apply_push(force)

            top_path = f"images/top_gt_{iteration}_{i}.png"
            side_path = f"images/side_gt_{iteration}_{i}.png"
            env.render_views(top_path, side_path)

            gpt_r, gt_r = gpt_reward_estimation(env, top_path, side_path, client)
            gt_rewards.append(gt_r)
            logs["gt"]["rewards"].append(gt_r)

        # === Evaluate GPT agent ===
        for i, a in enumerate(actions_gpt):
            a = np.clip(a, -force_clip, force_clip)
            force = [a[0], a[1], 0.0]

            env.reset()
            env.apply_push(force)

            top_path = f"images/top_gpt_{iteration}_{i}.png"
            side_path = f"images/side_gpt_{iteration}_{i}.png"
            env.render_views(top_path, side_path)

            gpt_r, gt_r = gpt_reward_estimation(env, top_path, side_path, client)
            gpt_rewards.append(gpt_r)
            logs["gpt"]["rewards"].append(gpt_r)

        # === REPS Updates ===
        def reps_update(actions, rewards):
            weights = np.exp((rewards - np.max(rewards)) / eta)
            weights /= weights.sum()
            new_mean = np.average(actions, axis=0, weights=weights)
            diffs = actions - new_mean
            new_cov = (weights[:, None, None] * (diffs[:, :, None] @ diffs[:, None, :])).sum(0)
            return new_mean, new_cov

        mean_gt, cov_gt = reps_update(actions_gt, np.array(gt_rewards))
        mean_gpt, cov_gpt = reps_update(actions_gpt, np.array(gpt_rewards))

        # === Logging ===
        logs["gt"]["avg_rewards"].append(np.mean(gt_rewards))
        logs["gt"]["cov_trace"].append(np.trace(cov_gt))
        logs["gt"]["mean_hist"].append(mean_gt.copy())

        logs["gpt"]["avg_rewards"].append(np.mean(gpt_rewards))
        logs["gpt"]["cov_trace"].append(np.trace(cov_gpt))
        logs["gpt"]["mean_hist"].append(mean_gpt.copy())

        # === Optional: plot and save ===
        def save_logs_and_plots():
            # 1. Rewards
            plt.figure()
            plt.plot(logs["gt"]["avg_rewards"], label="GT", color="green")
            plt.plot(logs["gpt"]["avg_rewards"], label="GPT", color="blue")
            plt.title(f"Average Reward at {iterations}")
            plt.xlabel("Iteration")
            plt.ylabel("Reward")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/avg_rewards.png")
            plt.close()

            # 2. Covariance trace
            plt.figure()
            plt.plot(logs["gt"]["cov_trace"], label="GT", color="green")
            plt.plot(logs["gpt"]["cov_trace"], label="GPT", color="blue")
            plt.title(f"Covariance Trace at {iterations}")
            plt.xlabel("Iteration")
            plt.ylabel("Trace")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/cov_trace.png")
            plt.close()

            # 3. Mean force
            gt_hist = np.array(logs["gt"]["mean_hist"])
            gpt_hist = np.array(logs["gpt"]["mean_hist"])

            plt.figure()
            plt.plot(gt_hist[:, 0], label="GT Fx", color="green")
            plt.plot(gpt_hist[:, 0], label="GPT Fx", color="blue")
            plt.plot(gt_hist[:, 1], label="GT Fy", linestyle='--', color="green")
            plt.plot(gpt_hist[:, 1], label="GPT Fy", linestyle='--', color="blue")
            plt.title(f"Mean Force at {iterations}")
            plt.xlabel("Iteration")
            plt.ylabel("Force")
            plt.legend()
            plt.grid(True)
            plt.savefig("plots/mean_force.png")
            plt.close()

            # 4. Correlation scatter
            plt.figure(figsize=(5, 5))
            plt.scatter(logs["gt"]["rewards"], logs["gpt"]["rewards"], alpha=0.6, edgecolor='k')
            plt.plot([-1, 1], [-1, 1], linestyle='--', color='red', label="y = x")
            plt.xlabel("GT Reward")
            plt.ylabel("GPT Reward")
            plt.xlim([-0.72, 0.02])
            plt.ylim([-0.72, 0.02])
            plt.title(f"Reward Correlation at {iterations}")
            plt.grid(True)
            plt.legend()
            plt.savefig("plots/correlation.png")
            plt.close()

            # Save raw logs
            for agent in ["gt", "gpt"]:
                np.savetxt(f"logs/{agent}_rewards_all.txt", np.array(logs[agent]["rewards"]))
                np.savetxt(f"logs/{agent}_avg_rewards.txt", np.array(logs[agent]["avg_rewards"]))
                np.savetxt(f"logs/{agent}_cov_trace.txt", np.array(logs[agent]["cov_trace"]))
                np.savetxt(f"logs/{agent}_mean_history.txt", np.array(logs[agent]["mean_hist"]))

        save_logs_and_plots()

    env.close()


if __name__ == "__main__":
    train()
