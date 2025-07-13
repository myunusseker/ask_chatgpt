import numpy as np
import matplotlib.pyplot as plt
import os
from slide_game import SlideToGoalEnv
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)
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

#---
#Example:
#If the cube is located at a corner of the top-down view, and the red goal circle is in the center, then the distance from the cube to the goal is approximately half of the diagonal of a 1m × 1m square. That’s about 0.7 meters, so the reward would be: -0.700.
#---
# === 1. Upload the image file ===
def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

# Dummy reward function (replace with GPT-based reward later)
def gpt_reward_estimation(env, top_img_path, side_img_path):
    final_pos = env.get_state()
    goal = env.goal_position
    gt_reward = -np.linalg.norm(final_pos[:2] - goal[:2])  # Negative distance as reward

    top_img_id = create_file(top_img_path)
    side_img_id = create_file(side_img_path)

    messages = [
        {"role": "user", "content": [
            {"type": "input_image", "file_id": top_img_id},
            {"type": "input_image", "file_id": side_img_id},
        ]}
    ]

    response = client.responses.create(
        model="gpt-4.1",
        instructions=system_prompt,
        input=messages,
    )
    print(response.output_text)
    gpt_reward = float(response.output_text)
    print(f"GPT reward is {gpt_reward}, GT reward is {gt_reward}")
    if gt_reward < -0.71:
        gpt_reward = -1
    return gpt_reward, gt_reward

# === Config ===
os.makedirs("images", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

action_dim = 2
batch_size = 20
n_iterations = 100
eta = 1.0
force_clip = 100.0

# Environment
env = SlideToGoalEnv(gui=False)

mean_gt = np.zeros(2)
mean_gpt = np.zeros(2)
cov_gt = np.eye(2) * 1000
cov_gpt = np.eye(2) * 1000

gt_rewards_all, gpt_rewards_all = [], []
gt_avg_rewards, gpt_avg_rewards = [], []
gt_cov_trace, gpt_cov_trace = [], []
gt_mean_hist, gpt_mean_hist = [], []

# === Training ===
for iteration in range(n_iterations):
    actions = np.random.multivariate_normal(mean_gt, cov_gt, batch_size)
    gt_rewards, gpt_rewards = [], []

    for i, a in enumerate(actions):
        a = np.clip(a, -force_clip, force_clip)
        force = [a[0], a[1], 0.0]

        env.reset()
        env.apply_push(force)

        top_path = f"images/top_{iteration}_{i}.png"
        side_path = f"images/side_{iteration}_{i}.png"
        env.render_views(top_path, side_path)

        gpt_r, gt_r = gpt_reward_estimation(env, top_path, side_path)
        gt_rewards.append(gt_r)
        gpt_rewards.append(gpt_r)
        gt_rewards_all.append(gt_r)
        gpt_rewards_all.append(gpt_r)

    def reps_update(actions, rewards):
        weights = np.exp((rewards - np.max(rewards)) / eta)
        weights /= weights.sum()
        new_mean = np.average(actions, axis=0, weights=weights)
        diffs = actions - new_mean
        new_cov = (weights[:, None, None] * (diffs[:, :, None] @ diffs[:, None, :])).sum(0)
        return new_mean, new_cov

    mean_gt, cov_gt = reps_update(actions, np.array(gt_rewards))
    gt_avg_rewards.append(np.mean(gt_rewards))
    gt_cov_trace.append(np.trace(cov_gt))
    gt_mean_hist.append(mean_gt.copy())

    mean_gpt, cov_gpt = reps_update(actions, np.array(gpt_rewards))
    gpt_avg_rewards.append(np.mean(gpt_rewards))
    gpt_cov_trace.append(np.trace(cov_gpt))
    gpt_mean_hist.append(mean_gpt.copy())

    # === Plotting per iteration ===
    gt_mean_hist_arr = np.array(gt_mean_hist)
    gpt_mean_hist_arr = np.array(gpt_mean_hist)

    # 1. Avg reward
    plt.figure()
    plt.plot(gt_avg_rewards, label="GT", color="green")
    plt.plot(gpt_avg_rewards, label="GPT", color="blue")
    plt.title("Average Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/avg_rewards_iter_{iteration}.png")
    plt.close()

    # 2. Cov trace
    plt.figure()
    plt.plot(gt_cov_trace, label="GT", color="green")
    plt.plot(gpt_cov_trace, label="GPT", color="blue")
    plt.title("Covariance Trace")
    plt.xlabel("Iteration")
    plt.ylabel("Trace")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/cov_trace_iter_{iteration}.png")
    plt.close()

    # 3. Mean force
    plt.figure()
    plt.plot(gt_mean_hist_arr[:, 0], label="GT Fx", color="green")
    plt.plot(gpt_mean_hist_arr[:, 0], label="GPT Fx", color="blue")
    plt.plot(gt_mean_hist_arr[:, 1], label="GT Fy", linestyle='--', color="green")
    plt.plot(gpt_mean_hist_arr[:, 1], label="GPT Fy", linestyle='--', color="blue")
    plt.title("Mean Force")
    plt.xlabel("Iteration")
    plt.ylabel("Force")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/mean_force_iter_{iteration}.png")
    plt.close()

    # 4. Correlation scatter
    gt_all = np.array(gt_rewards_all)
    gpt_all = np.array(gpt_rewards_all)

    plt.figure(figsize=(5, 5))
    plt.scatter(gt_all, gpt_all, alpha=0.6, edgecolor='k')
    plt.plot([-1, 1], [-1, 1], linestyle='--', color='red', label="y = x")
    plt.xlabel("GT Reward")
    plt.ylabel("GPT Reward")
    plt.xlim([-0.72, 0.02])
    plt.ylim([-0.72, 0.02])
    plt.title("Reward Correlation")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"plots/correlation_iter_{iteration}.png")
    plt.close()

    # Save logs after each iteration
    np.savetxt("logs/gt_rewards_all.txt", np.array(gt_rewards_all))
    np.savetxt("logs/gpt_rewards_all.txt", np.array(gpt_rewards_all))
    np.savetxt("logs/gt_avg_rewards.txt", np.array(gt_avg_rewards))
    np.savetxt("logs/gpt_avg_rewards.txt", np.array(gpt_avg_rewards))
    np.savetxt("logs/gt_cov_trace.txt", np.array(gt_cov_trace))
    np.savetxt("logs/gpt_cov_trace.txt", np.array(gpt_cov_trace))
    np.savetxt("logs/gt_mean_history.txt", np.array(gt_mean_hist))
    np.savetxt("logs/gpt_mean_history.txt", np.array(gpt_mean_hist))

env.close()