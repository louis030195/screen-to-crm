import argparse
import asyncio
import time
from PIL import ImageGrab, Image
import csv
import json
import requests
from transformers import FuyuProcessor, FuyuForCausalLM

# pylint: disable=pointless-string-statement
"""
This app monitors the user's screen during sales activities and updates a CRM system with the gathered information (CSV-based).

Steps:
1. Capture screen images.
2. Extract text from the images.
3. Batch the extracted text over several frames.
4. Process the text with a model to convert the user's actions into a CRM row. Update existing leads/accounts or create new ones if they don't exist.

Rules:
- Maintain two CSVs: one for leads and one for accounts (similar to LinkedIn Sales Navigator).
"""


def read_csv(file_path):
    """Read a CSV file and return its content as a list of dictionaries. Create the file if it does not exist."""
    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return [row for row in reader]
    except FileNotFoundError:
        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[])
            writer.writeheader()
        return []


def capture_screen():
    """Capture the screen and return the image."""
    im = ImageGrab.grab(bbox=None).convert(
        "RGB"
    )  # Capture the whole screen and ensure the image is in RGB format
    return im


async def on_activity(activity: str):
    """Trigger actions based on the activity text."""
    print(activity)


def build_prompt(leads, accounts):

    prompt = (
        f"You are an AI assistant that gets screenshots from a user screen doing sales and your job is to update a CRM system with the data extracted from the screen."
        f"Leads CSV content: {json.dumps(leads, indent=4)}\n\n"
        f"Accounts CSV content: {json.dumps(accounts, indent=4)}\n\n"
        "Return a JSON function call to update the CRM system with the processed text."
    )
    return prompt


async def main_loop(batch_size, sleep_interval, test_data_folder):
    """Main loop to capture screen, extract text, and process it with another model."""
    leads = read_csv("leads.csv")
    accounts = read_csv("accounts.csv")

    model_id = "adept/fuyu-8b"
    processor = FuyuProcessor.from_pretrained(model_id)
    model = FuyuForCausalLM.from_pretrained(model_id, device_map="mps")

    images = []

    if test_data_folder:
        import os
        from PIL import Image


        for filename in sorted(os.listdir(test_data_folder)):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(os.path.join(test_data_folder, filename)).convert(
                    "RGB"
                )
                images.append(image)

            if len(images) >= batch_size:
                prompt = build_prompt(leads, accounts)
                inputs = processor(text=prompt, images=images, return_tensors="pt").to(
                    "mps"
                )
                generation_output = model.generate(**inputs, max_new_tokens=500)
                processed_text = processor.batch_decode(
                    generation_output, skip_special_tokens=True
                )[0]
                await on_activity(processed_text)
                images = []
    else:
        while True:
            image = capture_screen()

            if len(images) >= batch_size:
                prompt = build_prompt(leads, accounts)
                inputs = processor(
                    text=prompt, images=images, return_tensors="pt"
                ).to("mps")
                generation_output = model.generate(**inputs, max_new_tokens=500)
                processed_text = processor.batch_decode(
                    generation_output, skip_special_tokens=True
                )[0]
                await on_activity(processed_text)
                images = []

            time.sleep(sleep_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the screen monitoring and CRM updating app."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of screen frames to batch before processing.",
    )
    parser.add_argument(
        "--sleep_interval",
        type=float,
        default=0.5,
        help="Time to sleep between capturing screen frames (in seconds).",
    )
    parser.add_argument(
        "--test_data_folder",
        default="test_data",
        type=str,
        help="Path to folder containing test data.",
    )

    args = parser.parse_args()
    asyncio.run(
        main_loop(
            args.batch_size,
            args.sleep_interval,
            args.test_data_file,
        )
    )
