import gradio as gr
import torch
import os
import zipfile
import requests
from TTS.api import TTS

# Set the environment variable for Coqui TTS
os.environ["COQUI_TOS_AGREED"] = "1"

# Initialize the device
device = "cuda"

# Initialize the TTS model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clone(text, url, language):
    # Download the zip file from HF
    response = requests.get(url)
    # Save the zip file to a temporary file
    with open("temp.zip", "wb") as f:
        f.write(response.content)
    # Extract the zip file to the current directory
    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()
    # Get the audio file paths
    audio_files = [f for f in os.listdir(".") if f.endswith(".wav")]
    # Use each audio file for voice cloning
    for audio_file in audio_files:
        tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path=f"./output_{audio_file}.wav")
    # Delete the temporary files
    for audio_file in audio_files:
        os.remove(audio_file)
    os.remove("temp.zip")
    # Return the output file paths
    return [f"./output_{audio_file}.wav" for audio_file in audio_files]

iface = gr.Interface(fn=clone,
                     inputs=["text", gr.components.Text(label="URL"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "ja", "zh-CN", "zh-TW"], label="Language")],
                     outputs=gr.Audio(type='filepath', multi=True),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )

                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))

iface.launch(share=True)
