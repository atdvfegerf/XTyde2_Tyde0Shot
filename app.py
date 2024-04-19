import gradio as gr
import torch
import os
import zipfile
import requests
from TTS.api import TTS
from pytube import YouTube

os.environ["COQUI_TOS_AGREED"] = "1"

device = "cuda"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def clone(text, url_or_text, language):
    if url_or_text.startswith("https://www.youtube.com/"):
        yt = YouTube(url_or_text)
        stream = yt.streams.filter(only_audio=True).first()
        stream.download(output_path=".", filename="temp")
        audio_file = "temp.mp4"
    else:
        audio_file = url_or_text

    tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path="./output.wav")

    if url_or_text.startswith("https://www.youtube.com/"):
        os.remove(audio_file)

    return "./output.wav"

iface = gr.Interface(fn=clone,
                     inputs=["text", gr.components.Text(label="URL or Text"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "ja", "zh-CN", "zh-TW"], label="Language")],
                     outputs=gr.Audio(type='filepath'),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )

                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))

iface.launch(share=True)
