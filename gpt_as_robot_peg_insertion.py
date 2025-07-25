from __future__ import annotations

import os
import json
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from environment_peg_insertion import PegInsertionEnvironment
from gpt_client import GPTClient
from responses.peg_insertion_response import ActionResponse
load_dotenv()

def initial_phase(env, client):
    env.apply_action([0.0, 0.0, 0.00], initial_env=True)
    env.render_views(image_postfix="initial", save_images=True)
    wrist_id = client.upload_image("images/wrist_initial.png")
    side_id = client.upload_image("images/side_initial.png")
    prompt_text = "This is the initialization phase. You are now in the peg insertion environment. There is no action applied yet. Whenever you propose an action, it will be applied to the robot, and after that the robot will try to insert the peg by bringing it down. The robot will stop if it hits any surfaces, and then you will observe the environment at that moment. The purpose of this step is just to get to know the environment and the objects. You can see the green peg. You can also see the red box on the table, and the square hole at the center of it. There is no need to propose any offset for this step. Just observe the environment. Please describe the current state of the environment and what you see in the images."
    content = [{"type": "input_image", "file_id": wrist_id},
              {"type": "input_image", "file_id": side_id},
              {"type": "input_text", "text": prompt_text}]
    messages = [{"role": "user", "content": content}]
    response, response_id = client.parse(messages, text_format=ActionResponse)
    print(response)
    return response_id

def discover_environment(env, client, discovery_steps=5, response_id=None):
    offset = [0.0, 0.0]

    for discover in range(0, discovery_steps+1):
        env.reset()
        env.apply_action([*offset, 0.00])
        env.render_views(image_postfix=f"discover_{discover}", save_images=True)
        side_id = client.upload_image(f"images/side_discover_{discover}.png")
        wrist_id = client.upload_image(f"images/wrist_discover_{discover}.png")

        content = []

        if discover == 0:
            content.append({"type": "input_text", "text": "This is the discovery phase. Before actually starting the task, you need to understand the environment and how the parameters work. You will have 5 attempts to discover the environment. In each attempt, you will be shown the resulting state of the environment. Use this information to understand how the parameters work."})
        
        content.append({"type": "input_text", "text": f"This is the outcome of action with offset values {offset}:"})
        content.append({"type": "input_image", "file_id": side_id})
        content.append({"type": "input_image", "file_id": wrist_id})

        if discover < discovery_steps:
            content.append({"type": "input_text", "text": f"Discovery phase: {discover + 1}/{discovery_steps}: Suggest an offset vector [OF1, OF2] to understand the environment."})
        else:
            content.append({"type": "input_text", "text": f"Discovery phase is over. Now let's start the actual task. The peg is currently reset at the initial position.\n Attempt 0: Suggest an offset vector [OF1, OF2] to insert the peg into the center hole of the red box."})

        messages = [{"role": "user","content": content}]

        response, response_id = client.parse(messages, previous_response_id=response_id if discover > 0 else None, text_format=ActionResponse)
        offset = response.offset
        reasoning = response.reasoning
        print(f"Discovery Mode {discover}/{discovery_steps}. GPT suggested offset: {offset}.\nThe reason is: {reasoning}.")
        
    return offset, response_id

def gpt_infer_action(
        env: PegInsertionEnvironment,
        client: GPTClient,
        max_attempts: int = 50,
        continuous: bool = False,
        discovery_phase: bool = False,
        discovery_steps: int = 5,
    ) -> tuple[list[float], float]:
    """
    Iteratively query GPT to infer the correct offset to apply to insert the peg.
    Returns (final_offset_vector)
    """
    os.makedirs("images/infer", exist_ok=True)
    env.reset()
    response_id = initial_phase(env, client)
    if discovery_phase:
        offset, response_id = discover_environment(env, client, discovery_steps=discovery_steps, response_id=response_id)

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt}")
        env.reset()
        env.apply_action([*offset, 0.00])
        env.render_views(image_postfix=f"attempt_{attempt}", save_images=True)

        wrist_id = client.upload_image(f"images/wrist_attempt_{attempt}.png")
        side_id = client.upload_image(f"images/side_attempt_{attempt}.png")

        content = []        

        content.append({"type": "input_text", "text": f"This is the outcome of action with offset values {offset}:"})
        content.append({"type": "input_image", "file_id": side_id})
        content.append({"type": "input_image", "file_id": wrist_id})
        content.append({"type": "input_text", "text": f"Investigate your previous force actions, reasonings and their resulting outcomes.\nAttempt {attempt + 1}/{max_attempts}: Based on previous attempts and their outcomes, suggest a better force to reach the red goal."})

        messages = [{"role": "user","content": content}]

        response, response_id = client.parse(messages, previous_response_id=response_id if attempt > 0 or discovery_phase else None, text_format=ActionResponse)
        offset = response.offset
        reasoning = response.reasoning
        print(f"Attempt {attempt + 1}/{max_attempts}: GPT suggested offset: {offset}.\nThe reason is: {reasoning}.")

    return offset

def main():
    continuous = False
    discovery_phase = True

    env = PegInsertionEnvironment(gui=False, hz=60)
    if continuous:
        system_prompt_path = "./system_prompts/peg_insertion_multi_step_action.txt"
    if not continuous:
        system_prompt_path = "./system_prompts/peg_insertion_one_step_action.txt"

    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(system_prompt)

    client = GPTClient(system_prompt=system_prompt, model_name="gpt-4.1",)

    final_offset = gpt_infer_action(
        env, 
        client, 
        continuous=continuous, 
        discovery_phase=discovery_phase
    )

    print(f"Final offset: {final_offset}")
    env.close()


if __name__ == "__main__":
    main()
