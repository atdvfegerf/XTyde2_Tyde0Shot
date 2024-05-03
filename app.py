import gradio as gr
import torch
import os
import zipfile
import requests
from TTS.api import TTS

# Set environment variable
os.environ["COQUI_TOS_AGREED"] = "1"

# Define constants
MODEL_PATH = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]

# Automatically detect and use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load TTS model
tts = TTS(MODEL_PATH).to(device)

def clone(text, url, language):
    # Download and extract audio file
    response = requests.get(url)
    with open("temp.zip", "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()
    audio_file = [f for f in os.listdir(".") if f.endswith(".wav")][0]

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
