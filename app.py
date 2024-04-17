import gradio as gr
import torch
import os
import zipfile
import requests
from TTS.api import TTS

os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clone_from_zip(text, url, language):
    response = requests.get(url)

    with open("temp.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()

    audio_files = [f for f in os.listdir(".") if f.endswith(".wav")]

    for audio_file in audio_files:
        tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path=f"./output_{audio_file}.wav")

    for audio_file in audio_files:
        os.remove(audio_file)

    os.remove("temp.zip")

    return [f"./output_{audio_file}.wav" for audio_file in audio_files]

def clone_from_files(text, audio_files, language):
    audio_files = [f for f in audio_files if f.endswith(".wav")]

    for audio_file in audio_files:
        tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path=f"./output_{audio_file}.wav")

    return [f"./output_{audio_file}.wav" for audio_file in audio_files]

def is_url(string):
    """
    Returns True if the string is a URL, False otherwise.
    """
    return string.startswith("http")

url = ""

iface = gr.Interface(fn=clone_from_zip if is_url(url) else clone_from_files,
                     inputs=["text", gr.components.File(type="filepath", label="Audio Files"), gr.Text(label="URL", placeholder="Enter URL to zip file"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "ja", "zh-CN", "zh-TW"], label="Language")],
                     outputs=gr.Audio(type='filepath', label='Synthesized Audio'),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )

                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))

iface.launch(share=True)
