from __future__ import annotations

import os
import json
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from environment import SlideToGoalEnv
from gpt_client import GPTClient
from responses.action_response import ActionResponse
load_dotenv()


def parse_parameter_understanding(response: str) -> list[float]:
    try:
        data = json.loads(response)
        return data["parameter_understanding"]
    except:
        raise ValueError(f"Failed to parse JSON parameter_understanding from: {response}")

def gpt_infer_action(
        env: SlideToGoalEnv, 
        client: GPTClient, 
        max_attempts: int = 50, 
        continuous: bool = False,
        single_view: bool = False,
        view: str = "top",
        discovery_phase: bool = False,
        discovery_steps: int = 5,
    ) -> tuple[list[float], float]:
    """
    Iteratively query GPT to infer the correct force to apply to reach the goal.
    Returns (final_force_vector, final_reward)
    """
    os.makedirs("images/infer", exist_ok=True)
    env.reset()

    force = [0.0, 0.0]
    reward = -np.inf

    if discovery_phase:
        for discover in range(discovery_steps):
            if not continuous:
                env.reset()

            env.apply_push([*force, 0.00])

            top_path = f"images/infer/top_discover_{discover}.png"
            side_path = f"images/infer/side_discover_{discover}.png"
            env.render_views(top_path, side_path, use_static_side=True)

            if not single_view:
                top_id = client.upload_image(top_path)
                side_id = client.upload_image(side_path)
            else:
                if view == "top":
                    top_id = client.upload_image(top_path)
                elif view == "side":
                    side_id = client.upload_image(side_path)
                else:
                    raise ValueError(f"Invalid view: {view}. Choose 'top' or 'side'.")

            if discover == 0:
                prompt_text = "Before actually starting the task, you need to understand the environment and how the parameters work. You will have 5 attempts to discover the environment. In each attempt, you will be shown the resulting state of the environment. Use this information to understand how the parameters work.\n Discovery attempt: 1/5. Suggest a force vector [F1, F2] to understand the environment."
            else:
                prompt_text = f"Discovery attempt: {discover + 1}/5. Suggest a force vector [F1, F2] to understand the environment."
            
            content = []
            if not single_view:
                content.append({"type": "input_image", "file_id": side_id})
                content.append({"type": "input_image", "file_id": top_id})
            else:
                content.append({"type": "input_image", "file_id": top_id if view == "top" else side_id})
                
            content.append({"type": "input_text", "text": prompt_text})

            messages = [{"role": "user","content": content}]

            response, previous_response_id = client.parse(messages, previous_response_id=previous_response_id if discover > 0 else None, text_format=ActionResponse)
            force = response.force
            reasoning = response.reasoning
            print(f"Discovery Mode. GPT suggested force: {force}.\nThe reason is: {reasoning}.")

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt}")

        # Apply the last suggested force and observe result
        if not continuous:
            env.reset()
        env.apply_push([*force, 0.00])

        top_path = f"images/infer/top_attempt_{attempt}.png"
        side_path = f"images/infer/side_attempt_{attempt}.png"
        env.render_views(top_path, side_path, use_static_side=True)

        # Evaluate current state
        final_pos = env.get_state()
        goal = env.goal_position
        reward = -np.linalg.norm(final_pos[:2] - goal[:2])
        print(f"Resulting reward: {reward:.3f}")

        # Done if the est   imation is successful
        if reward > -0.05:
            break

        # Otherwise Upload images
        if not single_view:
            top_id = client.upload_image(top_path)
            side_id = client.upload_image(side_path)
        else:
            if view == "top":
                top_id = client.upload_image(top_path)
            elif view == "side":
                side_id = client.upload_image(side_path)
            else:
                raise ValueError(f"Invalid view: {view}. Choose 'top' or 'side'.")

        # Construct prompt text and message
        if attempt == 0:
            if continuous:
                prompt_text = "Now let's start the actual task. Suggest a force vector [F1, F2] to push it to the center of the red goal."
            else:
                prompt_text = "Now let's start the actual task. The cube is currently reset at the initial position. Suggest a force vector [F1, F2] to push it to the center of the red goal."
        else:
            prompt_text = f"Investigate your previous force actions, reasonings and their resulting outcomes. Based on those, suggest a better force to reach the red goal."

        content = []
        if not single_view:
            content.append({"type": "input_image", "file_id": side_id})
            content.append({"type": "input_image", "file_id": top_id})
        else:
            content.append({"type": "input_image", "file_id": top_id if view == "top" else side_id})
        
        content.append({"type": "input_text", "text": prompt_text})

        messages = [{"role": "user","content": content}]

        response, previous_response_id = client.parse(messages, previous_response_id=previous_response_id if attempt > 0 or discovery_phase else None, text_format=ActionResponse)
        force = response.force
        reasoning = response.reasoning
        print(f"GPT suggested force: {force}.\nThe reason is: {reasoning}.")

    return force, reward

def main():
    continuous = False
    single_view = True
    discovery_phase = True
    view = "top" # Choose between "top" or "side"

    env = SlideToGoalEnv(gui=True, speed=120)
    if continuous and not single_view:
        system_prompt_path = "./system_prompts/cube_multi_step_action.txt"
    if not continuous and not single_view:
        system_prompt_path = "./system_prompts/cube_one_step_action.txt"
    if not continuous and single_view:
        system_prompt_path = "./system_prompts/cube_one_step_action_single_view.txt"
    if continuous and single_view:
        system_prompt_path = "./system_prompts/cube_multi_step_action_single_view.txt"

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(system_prompt)

    client = GPTClient(system_prompt=system_prompt, model_name="gpt-4.1",)

    final_force, final_reward = gpt_infer_action(
        env, 
        client, 
        continuous=continuous, 
        single_view=single_view, 
        view=view,
        discovery_phase=discovery_phase
    )

    print(f"Final reward: {final_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
