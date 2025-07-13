from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# === 1. Upload the image file ===
def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


# Provide your image path here
image_path = "6.jpeg"
file_id = create_file(image_path)

# === 2. System Prompt (Robot Brain Role) ===
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

# === 3. User Input: task + parameters + image ===
task_description = "Scrape the pan with the spatula."
parameters_to_estimate = ["force_value"]
task_description = "Pick a spatula type to scrape the shown grease/dirt out of the pan. Available spatula types are: silicone, wooden, metal"
parameters_to_estimate = ["spatula_type","force_value"]

# Combine into messages
messages = [
    {"role": "user", "content": [
        {"type": "input_text", "text": f"Task: {task_description}"},
        {"type": "input_text", "text": "Estimate the following parameters: " + ", ".join(parameters_to_estimate)},
        {"type": "input_image", "file_id": file_id},
    ]}
]

# === 4. Call GPT ===

MODEL_NAME = "gpt-4.1"
response = client.responses.create(
    model=MODEL_NAME,
    instructions=system_prompt,
    input=messages
)

# === 5. Output Response ===
output_text = response.output_text
print(f"\n--- {MODEL_NAME} Output ---\n")
print(output_text)

# === Optional: Save output to a JSON file (if valid JSON is returned) ===
try:
    parsed_json = json.loads(output_text)
    with open("robotic_parameter_output.json", "w") as f:
        json.dump(parsed_json, f, indent=2)
    print("\n✅ Output saved to robotic_parameter_output.json")
except json.JSONDecodeError:
    print("\n⚠️ Warning: Output is not valid JSON. Please check formatting.")
