from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from environment import SlideToGoalEnv
from gpt_client import GPTClient

system_prompt = """
You are given two rendered images of a game environment: one from a side view and one from a top-down view. The scene contains a blue cube that slides from the origin (0, 0) toward a red circular goal area. The ground is made of 1-meter-by-1-meter tiles for scale reference. The red circular goal has a radius of 0.100 meters.

Your task is to suggest a force vector [Fx, Fy] that, when applied to the cube, moves it as close as possible onto the center of the red goal. The force is applied horizontally, and you should not suggest any vertical force (Fz = 0).

After each suggestion, you will be shown the result (top and side view images), and you may adjust your force recommendation accordingly. After you adjust the force recommendation, the environment will be reset and the new adjusted force will be applied from the origin.

Fx and Fy values can be in the range of (-100.00,100.00)

Always respond with a JSON object in the following format:
{
  "reasoning": "<short explanation of how the cube moved and how you plan to change the force>",
  "force": [Fx, Fy]
}"""

system_prompt_cont = """
You are given two pairs of rendered images of a game environment: the previous state and the current state, each with one side view and one top-down view. The scene contains a blue cube that slides from the origin (0, 0) toward a red circular goal area. The ground is made of 1-meter-by-1-meter tiles for scale reference. The red circular goal has a radius of 0.100 meters.

Your task is to suggest a force vector [Fx, Fy] that, when applied to the cube, will continue guiding it closer to the red goal. The cube keeps moving from its current place based on each new force you suggest. It is not reset between steps.

Fx and Fy values can be in the range of (-100.00,100.00)

Always respond with a JSON object in the following format:
{
  "reasoning": "<short explanation of how the cube moved and how you plan to change the force>",
  "force": [Fx, Fy]
}"""


def parse_force(response: str) -> list[float]:
    try:
        data = json.loads(response)
        return data["force"]
    except:
        raise ValueError(f"Failed to parse JSON force vector from: {response}")


def parse_reasoning(response: str) -> str:
    try:
        data = json.loads(response)
        return data["reasoning"]
    except:
        raise ValueError(f"Failed to parse JSON reasoning from: {response}")


def gpt_infer_action(env: SlideToGoalEnv, client: GPTClient, max_attempts: int = 50) -> tuple[list[float], float]:
    """
    Iteratively query GPT to infer the correct force to apply to reach the goal.
    Returns (final_force_vector, final_reward)
    """
    os.makedirs("images/infer", exist_ok=True)

    force = [0.0, 0.0]
    reward = -np.inf
    previous_reasoning = ""

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt}")

        # Apply the last suggested force and observe result
        env.reset()
        env.apply_push([*force, 0.00])

        top_path = f"images/infer/top_attempt_{attempt}.png"
        side_path = f"images/infer/side_attempt_{attempt}.png"
        env.render_views(top_path, side_path)

        # Evaluate reward after GPT's new suggestion will be applied next loop
        final_pos = env.get_state()
        goal = env.goal_position
        reward = -np.linalg.norm(final_pos[:2] - goal[:2])
        print(f"Resulting reward: {reward:.3f}")

        # Done if the estimation is successful
        if reward > -0.05:
            break

        # Otherwise Upload images
        top_id = client.upload_image(top_path)
        side_id = client.upload_image(side_path)

        # Construct prompt text and message
        if attempt == 0:
            prompt_text = "The cube is currently at the origin. Suggest a force vector [Fx, Fy] to push it to the center of the red goal."
        else:
            prompt_text = f"Last applied force was {force}. Your previous reasoning was: \"{previous_reasoning}\" Suggest a better force to reach the red goal."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": side_id},
                    {"type": "input_image", "file_id": top_id},
                    {"type": "input_text", "text": prompt_text}
                ]
            }
        ]

        response = client.chat(system_prompt, messages)
        previous_reasoning = parse_reasoning(response)
        force = parse_force(response)
        print(f"GPT suggested force: {force}. The reason is: {previous_reasoning}")

    return force, reward

def gpt_infer_action_continuous(env: SlideToGoalEnv, client: GPTClient, max_attempts: int = 50) -> float:
    os.makedirs("images/infer_continuous", exist_ok=True)

    env.reset()
    force = [0.0, 0.0]
    reward = -np.inf
    reasoning_history = []
    force_history = []
    top_history = []
    side_history = []

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt}")

        env.apply_push([*force, 0.00])

        top_path = f"images/infer_continuous/top_attempt_{attempt}.png"
        side_path = f"images/infer_continuous/side_attempt_{attempt}.png"
        env.render_views(top_path, side_path)

        final_pos = env.get_state()
        goal = env.goal_position
        reward = -np.linalg.norm(final_pos[:2] - goal[:2])
        print(f"Resulting reward: {reward:.3f}")

        if reward > -0.05:
            break

        top_id = client.upload_image(top_path)
        side_id = client.upload_image(side_path)

        prompt_images = [
            {"type": "input_image", "file_id": side_id},
            {"type": "input_image", "file_id": top_id},
        ]

        if attempt > 0:
            prev_top_id = client.upload_image(top_history[-1])
            prev_side_id = client.upload_image(side_history[-1])
            prompt_images.insert(0, {"type": "input_image", "file_id": prev_top_id})
            prompt_images.insert(0, {"type": "input_image", "file_id": prev_side_id})

        if attempt == 0:
            prompt_text = "The cube starts at the origin. Suggest a force vector [Fx, Fy] to start moving it toward the red goal."
        else:
            prompt_text = f"Last applied force was {force}. Your previous reasoning was: \"{reasoning_history[-1]}\". Based on the change between previous and current images, suggest the next force."

        messages = [
            {
                "role": "user",
                "content": prompt_images + [
                    {"type": "input_text", "text": prompt_text}
                ]
            }
        ]

        response = client.chat(system_prompt_cont, messages)
        reasoning = parse_reasoning(response)
        force = parse_force(response)

        reasoning_history.append(reasoning)
        force_history.append(force)
        top_history.append(top_path)
        side_history.append(side_path)

        print(f"GPT suggested force: {force}. The reason is: {reasoning}")

    return reward


def main():
    env = SlideToGoalEnv(gui=True, speed=120)
    client = GPTClient()

    # You can switch between the reset-based and continuous version here:
    final_force, final_reward = gpt_infer_action(env, client)
    #final_reward = gpt_infer_action_continuous(env, client)

    print(f"Final reward: {final_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
