import gradio as gr
import os
import requests
import torch
import zipfile
from TTS.api import TTS
from pydub import AudioSegment

#download for mecab
os.system('python -m unidic download')


os.environ["COQUI_TOS_AGREED"] = "1"

MODEL_PATH = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".mp4"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tts = TTS(MODEL_PATH).to(device)

def download_audio_file(url):
    try:
        response = requests.get(url, stream=True)
        file_name = url.split("/")[-1]
        file_extension = os.path.splitext(file_name)[-1].lower()
        if file_extension not in AUDIO_FORMATS:
            raise ValueError(f"Invalid audio file format: {file_extension}")
        with open(file_name, "wb") as f:
            f.write(response.content)  # Write the entire response content at once
        return file_name
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio file: {e}")
        return None

def convert_to_wav(input_audio_file):
    file_extension = os.path.splitext(input_audio_file)[-1].lower()
    if file_extension!= ".wav":
        audio = AudioSegment.from_file(input_audio_file)
        audio.export("temp.wav", format="wav")
        os.remove(input_audio_file)
        return "temp.wav"
    return input_audio_file

def synthesize_text(text, input_audio_file, language):
    input_audio_file = convert_to_wav(input_audio_file)
    tts.tts_to_file(text=text, speaker_wav=input_audio_file, language=language, file_path="./output.wav")
    return "./output.wav"

def clone(text, input_file, language, url=None, use_url=False):
    if use_url:
        if url is None:
            return None
        input_audio_file = download_audio_file(url)
        if input_audio_file is None:
            return None
    else:
        if input_file is None:
            return None
        input_audio_file = input_file.name

    output_file_path = synthesize_text(text, input_audio_file, language)
    return output_file_path

iface = gr.Interface(
    fn=clone,
    inputs=["text", gr.File(label="Input File", file_types=AUDIO_FORMATS), gr.Dropdown(choices=LANGUAGES, label="Language"), gr.Text(label="URL"), gr.Checkbox(label="Use URL", value=False)],
    outputs=gr.Audio(type='filepath'),
    title='Voice Clone',
    description=""" by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ ) use this colab with caution <3. """,
    theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate")
)

iface.launch(share=True)
