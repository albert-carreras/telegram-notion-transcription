import os
import sys
import asyncio

from datetime import datetime, time
from contextlib import contextmanager

import whisperx
import requests

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters
from notion_client import Client
from notion_client.errors import APIResponseError
from dotenv import load_dotenv

REMINDER_TIME = time(hour=19, minute=59)

load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1
DEVICE = "cpu"
AUDIO_FILE = "input.wav"
BATCH_SIZE = 32
COMPUTE_TYPE = "int8"
MAX_NOTION_BLOCK_LENGTH = 2000

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOTION_JOURNAL_PAGE_ID = os.getenv("NOTION_JOURNAL_PAGE_ID")
client = OpenAI(
    api_key=OPENAI_API_KEY,
)


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


print("Loading WhisperX model...")
with suppress_stdout_stderr():
    whisper_model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="en", threads=12)

notion = Client(auth=NOTION_TOKEN)


async def send_daily_reminder(context: ContextTypes.DEFAULT_TYPE):
    message = "🌟 Daily Reminder"

    updates = await context.bot.get_updates()

    unique_chat_ids = set(update.message.chat_id for update in updates if update.message)

    for chat_id in unique_chat_ids:
        await context.bot.send_message(chat_id=chat_id, text=message)


def transcribe_audio(audio_file):
    audio = whisperx.load_audio(audio_file)
    print("--- Transcribing ---")
    with suppress_stdout_stderr():
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)

    if result["segments"]:
        return "\n".join(segment["text"] for segment in result["segments"])
    else:
        return "No speech detected."


def create_rich_text_blocks(text):
    chunks = [text[i:i + MAX_NOTION_BLOCK_LENGTH] for i in range(0, len(text), MAX_NOTION_BLOCK_LENGTH)]
    return [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}]
            }
        } for chunk in chunks
    ]


def save_to_notion(transcription, raw, image_url=None):
    now = datetime.now()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an assistant that generates concise, meaningful titles for journal entries based on their content. It's a journal entry and it will be on a list of many other journal entries, therefore make the title recognizable, don't use words that can be often used with journal entries as this will make the titles repetitive."},
            {"role": "user", "content": transcription}
        ]
    )
    title = response.choices[0].message.content.strip('\"')

    response_title = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an assistant that summarizes journal entries in a couple of sentences maximum. The summary is addressed to them, the user. You just summarize the text, you don't add any other comments to it."},
            {"role": "user", "content": transcription}
        ]
    )
    summary = response_title.choices[0].message.content

    response_evaluation = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an assistant that evaluates journal entries and gives a sentiment analysis. You just return a number representing your evaluation of the sentiment analysis. You use a scale from 1 to 100 on how positive, happy and optimistic the journal entry is. For reference, a 1 would be when something terrible happened, like a death. A 100 would be when something wonderful happened, like the birth of your first child. A 50 would be a normal, average day, routine, nothing special happened."},
            {"role": "user", "content": transcription}
        ]
    )
    sentiment = response_evaluation.choices[0].message.content.strip('\"')

    try:
        new_page = notion.pages.create(
            parent={"database_id": NOTION_JOURNAL_PAGE_ID},
            properties={
                "Title": {"type": "title",
                          "title": [{"type": "text", "text": {"content": title}}]},
                "Date": {
                    "type": "date",
                    "date": {"start": now.isoformat(), "end": None}
                },
                "Sentiment": {
                    "type": "number",
                    "number": int(sentiment)
                },
            },
            children=[
                *create_rich_text_blocks(transcription),
                {
                    "object": "block",
                    "type": "quote",
                    "quote": {
                        "rich_text": [{
                            "annotations": {
                                "italic": True,
                            },
                            "type": "text",
                            "text": {"content": summary}
                        }]
                    }
                },
                {
                    "object": "block",
                    "type": "toggle",
                    "toggle": {
                        "rich_text": [{"type": "text", "text": {"content": "Raw Data"}}],
                        "children": create_rich_text_blocks(raw),
                    },
                },
            ]
        )

        if image_url:
            notion.blocks.children.append(
                block_id=new_page['id'],
                children=[
                    {
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {
                                "url": image_url
                            }
                        }
                    }
                ]
            )

        print(f"Journal entry saved to Notion page: {new_page['url']}")
        return True, new_page['url']

    except APIResponseError as e:
        error_message = f"Error saving journal entry to Notion: {str(e)}"
        print(error_message)
        return False, error_message


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    await update.message.reply_text("Transcribing...")

    if message.voice:
        try:
            file = await context.bot.get_file(message.voice.file_id)
            try:
                response = requests.get(file.file_path, timeout=30)
                response.raise_for_status()

                with open("input.wav", 'wb') as f:
                    f.write(response.content)

            except Exception as e:
                await update.message.reply_text(f"Error downloading file 1: {e}")
                return
        except Exception as e:
            await update.message.reply_text(f"Error downloading file 2: {e}")
            return

        try:
            transcription = transcribe_audio(AUDIO_FILE)
        except Exception as e:
            await update.message.reply_text(f"Error transcribing audio: {e}")
            return

        try:
            raw = transcription
            transcription = cleanup_with_gpt4o_mini(transcription)
        except Exception as e:
            await update.message.reply_text(f"Error cleaning up text with GPT-4o mini: {e}")
            return

        success, url = save_to_notion(transcription, raw)

        if success:
            await update.message.reply_text(f"Saved: {url}")
        else:
            await update.message.reply_text("Error saving to Notion. Please check your settings.")
    else:
        await update.message.reply_text("No voice message detected.")


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    await update.message.reply_text("Processing image...")

    if message.photo:
        try:
            file = await context.bot.get_file(message.photo[-1].file_id)
            image_url = file.file_path

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image as if it were a concise journal entry. Keep it to two or three sentences maximum."},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
            )
            caption = response.choices[0].message.content

            success, url = save_to_notion(caption, "Image caption", image_url)

            if success:
                await update.message.reply_text(f"Image processed and saved: {url}")
            else:
                await update.message.reply_text("Error saving to Notion. Please check your settings.")
        except Exception as e:
            await update.message.reply_text(f"Error processing image: {e}")
    else:
        await update.message.reply_text("No image detected.")


def cleanup_with_gpt4o_mini(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an assistant that proofreads and cleans up text. You may only correct grammar, spelling, and punctuation, but you will not change the meaning of the text whatsoever. Only respond with the cleaned-up text."},
            {"role": "user", "content": text}
        ]
    )
    cleaned_text = response.choices[0].message.content
    return cleaned_text


async def start(update):
    await update.message.reply_text(
        "Welcome! Send me a voice message and I'll transcribe it and save it to your Notion journal.")

def main():
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .enable_job_queue()
        .build()
    )


    application.job_queue.run_daily(
        send_daily_reminder,
        REMINDER_TIME,
        days=(0, 1, 2, 3, 4, 5, 6)
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot is running. Send a voice message to transcribe and save to Notion.")
    application.run_polling()


if __name__ == "__main__":
    main()
