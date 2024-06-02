import argparse
import asyncio
import time
from PIL import ImageGrab, Image
import csv
import pytesseract
import json
from mlc_llm import MLCEngine

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


def process_image_with_tesseract(image):
    """Process the captured image and generate text using Tesseract."""
    text = pytesseract.image_to_string(image)
    return text


async def on_activity(activity: str):
    """Trigger actions based on the activity text."""
    # if "urgent" in activity:
    #     await send_alert(activity)
    # elif "meeting" in activity:
    #     await schedule_meeting(activity)
    # update_crm(activity)
    print(activity)


async def send_alert(activity):
    """Send an alert based on the activity."""
    print(f"Alert: {activity}")


async def schedule_meeting(activity):
    """Schedule a meeting based on the activity."""
    print(f"Scheduling meeting: {activity}")


def process_text_with_logic_processor(text, logic_processor_model):
    """Process the extracted text using a logic processor model."""
    inputs = logic_processor_model.tokenizer(text, return_tensors="pt").to("mps")
    generation_args = {
        "max_new_tokens": 500,
        "temperature": 0.7,
        "do_sample": True,
    }
    print("Processing text with logic processor...")
    generate_ids = logic_processor_model.generate(
        **inputs,
        eos_token_id=logic_processor_model.tokenizer.eos_token_id,
        **generation_args,
    )
    response = logic_processor_model.tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


def build_prompt(screen_texts, leads, accounts, batch_size):
    """Build the prompt for the logic processor."""
    batched_text = {
        "frames": [
            {"frame_number": i + 1, "text": text} for i, text in enumerate(screen_texts)
        ]
    }
    batched_text = json.dumps(batched_text, indent=4)
    prompt = (
        f"You are an AI assistant that get text from a user screen doing sales (through OCR) and you job is to update a CRM system with the data extracted from the screen."
        f"This is the extracted text on the user's screen over the past {batch_size} frames: \n\n{batched_text}\n\n"
        f"Leads CSV content: {json.dumps(leads, indent=4)}\n\n"
        f"Accounts CSV content: {json.dumps(accounts, indent=4)}\n\n"
        "Return a JSON function call to update the CRM system with the processed text."
    )
    return prompt


async def main_loop(batch_size, sleep_interval, test_data_file):
    """Main loop to capture screen, extract text, and process it with another model."""
    leads = read_csv("leads.csv")
    accounts = read_csv("accounts.csv")

    logic_processor_model = MLCEngine("HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC")

    screen_texts = []

    if test_data_file:
        with open(test_data_file, "r") as file:
            test_data = json.load(file)
        for entry in test_data["frames"]:
            screen_texts.append(entry["text"])
            if len(screen_texts) >= batch_size:
                prompt = build_prompt(screen_texts, leads, accounts, batch_size)
                response = logic_processor_model.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
                    stream=False,
                )
                processed_text = response.choices[0].message.content
                await on_activity(processed_text)
                screen_texts = []
    else:
        while True:
            image = capture_screen()
            text = process_image_with_tesseract(image)
            screen_texts.append(text)

            if len(screen_texts) >= batch_size:
                prompt = build_prompt(screen_texts, leads, accounts, batch_size)
                response = logic_processor_model.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
                    stream=False,
                )
                processed_text = response.choices[0].message.content
                await on_activity(processed_text)
                screen_texts = []

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
        "--test_data_file",
        default="test_data.json",
        type=str,
        help="Path to JSON file containing test data.",
    )

    args = parser.parse_args()
    asyncio.run(
        main_loop(
            args.batch_size,
            args.sleep_interval,
            args.test_data_file,
        )
    )
