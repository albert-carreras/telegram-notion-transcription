import os
import sys
from datetime import datetime
from contextlib import contextmanager

import whisperx
import requests

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, ContextTypes, CommandHandler, MessageHandler, filters
from notion_client import Client
from notion_client.errors import APIResponseError

# Constants
SAMPLE_RATE = 24000
CHANNELS = 1
DEVICE = "cpu"
AUDIO_FILE = "input.wav"
BATCH_SIZE = 32
COMPUTE_TYPE = "int8"

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
    whisper_model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="en", threads=8)

notion = Client(auth=NOTION_TOKEN)


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
    chunks = [text[i:i+MAX_NOTION_BLOCK_LENGTH] for i in range(0, len(text), MAX_NOTION_BLOCK_LENGTH)]
    return [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": chunk}}]
            }
        } for chunk in chunks
    ]


def save_to_notion(transcription, raw):
    now = datetime.now()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that generates concise, meaningful titles for journal entries based on their content."},
            {"role": "user", "content": transcription}
        ]
    )
    title = response.choices[0].message.content.strip('\"')

    response_title = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes journal entries in a couple of sentences maximum. The summary is addressed to them, the user. You just summarize the text, you don't add any other comments to it."},
            {"role": "user", "content": transcription}
        ]
    )
    summary = response_title.choices[0].message.content

    response_evaluation = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that evaluates journal entries and gives a sentiment analysis. You just return a number representing your evaluation of the sentiment analysis. You use a scale from 1 to 100 on how positive, happy and optimistic the journal entry is. For reference, a 1 would be when something terrible happened, like a death. A 100 would be when something wonderful happened, like the birth of your first child. A 50 would be a normal, average day, routine, nothing special happened."},
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
                response = requests.get(file.file_path, timeout=30)  # 30 seconds timeout
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


def cleanup_with_gpt4o_mini(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an assistant that proofreads and cleans up text. You may correct grammar, spelling, and punctuation, but you will not change the meaning of the text. Only respond with the cleaned-up text."},
            {"role": "user", "content": text}
        ]
    )
    cleaned_text = response.choices[0].message.content
    return cleaned_text


async def start(update, context):
    await update.message.reply_text(
        "Welcome! Send me a voice message and I'll transcribe it and save it to your Notion journal.")


def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))

    print("Bot is running. Send a voice message to transcribe and save to Notion.")
    application.run_polling()


if __name__ == "__main__":
    main()