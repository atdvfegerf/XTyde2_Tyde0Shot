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

def clone(text, url, language):
    response = requests.get(url)

    with open("temp.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()

    audio_file = [f for f in os.listdir(".") if f.endswith(".wav")][0]

    tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path="./output.wav")

    os.remove(audio_file)
    os.remove("temp.zip")

    return "./output.wav"
    
    def download_video(youtube_url, save_path):
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=save_path)
        return f"Video scaricato correttamente in {save_path}"
    except Exception as e:
        return f"Si è verificato un errore durante il download: {str(e)}"

def video_downloader(youtube_url, save_path):
    return download_video(youtube_url, save_path)

iface = gr.Interface(fn=clone,
                     inputs=["text", gr.components.Text(label="URL"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "ja", "zh-CN", "zh-TW"], label="Language")],
                     outputs=gr.Audio(type='filepath'),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )

                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))
fn=video_downloader,
    inputs=["text", "text"],
    outputs="text",
    title="YouTube Video Downloader",
    description="Inserisci l'URL di un video di YouTube e la directory di destinazione per il download.",
    article="This model downloads YouTube videos given their URL and destination path.",
    example=[
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "/content/"]


iface.launch(share=True)
