import gradio as gr
import torch
import os
import zipfile
import requests
from TTS.api import TTS
from pydub import AudioSegment

# Set environment variable
os.environ["COQUI_TOS_AGREED"] = "1"

# Define constants
MODEL_PATH = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".mp4"]

# Automatically detect and use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load TTS model
tts = TTS(MODEL_PATH).to(device)

def convert_to_wav(audio_file):
    # Extract file extension
    file_extension = os.path.splitext(audio_file)[-1].lower()

    # Convert audio file to .wav format
    if file_extension != ".wav":
        audio = AudioSegment.from_file(audio_file)
        audio.export("temp.wav", format="wav")
        os.remove(audio_file)
        audio_file = "temp.wav"

    return audio_file

def clone(text, url, language):
    # Download zip file
    response = requests.get(url)
    with open("temp.zip", "wb") as f:
        f.write(response.content)

    # Extract audio file from zip archive
    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if os.path.splitext(file)[-1].lower() in AUDIO_FORMATS:
                zip_ref.extract(file, ".")
                audio_file = file
                break

    # Convert audio file to .wav format
    audio_file = convert_to_wav(audio_file)

    # Generate audio using TTS model
    tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path="./output.wav")

    # Clean up
    os.remove(audio_file)
    os.remove("temp.zip")

    return "./output.wav"

# Create Gradio interface
iface = gr.Interface(
    fn=clone,
    inputs=["text", gr.components.Text(label="URL"), gr.Dropdown(choices=LANGUAGES, label="Language")],
    outputs=gr.Audio(type='filepath'),
    title='Voice Clone',
    description=""" by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ ) use this colab with caution <3. """,
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate")
)

# Launch Gradio interface
iface.launch(share=True)
