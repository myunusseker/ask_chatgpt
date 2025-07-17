from typing import List
from gpt_client import GPTClient

system_prompt = """
You are the reasoning core of a real-world robot, powered by GPT-4o. At specific decision points during physical manipulation tasks, you will be asked to estimate parameters required to complete the current subtask.

You will be given:
- A natural language task description.
- A scene description or image.
- A list of specific parameters to estimate (e.g., force in newtons, alignment in cm, grip type).
- Optionally, robot end-effector state and/or history.

Your responsibilities:
1. Analyze the task and the scene input.
2. Estimate each requested parameter using concrete, real-world values with appropriate physical units (e.g., "8.5 N", "6.2 cm", "microfiber brush").
3. Provide a grounded explanation (reasoning) for each parameter based on the provided input.
4. Assign a confidence level for each estimate: "high", "medium", or "low".

Use this JSON format exactly:
{
  "parameters": {
    "parameter_name": {
      "value": "actual_value_with_unit",
      "reasoning": "clear and grounded explanation",
      "confidence": "high | medium | low"
    },
    ...
  }
}

Always fill in all fields for each parameter. Base your reasoning only on the provided input. Use physical plausibility and common-sense priors, but ground decisions in the context.
"""

def estimate_parameters(image_path: str, task_description: str, parameters: List[str]) -> None:
    client = GPTClient()
    file_id = client.upload_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"Task: {task_description}"},
                {"type": "input_text", "text": "Estimate the following parameters: " + ", ".join(parameters)},
                {"type": "input_image", "file_id": file_id},
            ],
        }
    ]

    output = client.chat(system_prompt, messages)
    print(output)
    client.save_json(output, "robotic_parameter_output.json")


if __name__ == "__main__":
    estimate_parameters(
        image_path="assests/6.jpeg",
        task_description="Pick a spatula type to scrape the shown grease/dirt out of the pan. Available spatula types are: silicone, wooden, metal",
        parameters=["spatula_type", "force_value"],
    )