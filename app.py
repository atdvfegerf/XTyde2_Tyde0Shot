import subprocess
import os
import torch
import urllib.request
import librosa
from moviepy.editor import VideoFileClip
from TTS.api import TTS

# Set the device to CPU
device = "cpu"

os.environ["COQUI_TOS_AGREED"] = "1"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def convert_audio_to_wav(file_path):
    """Convert the given audio file to WAV format."""
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext == ".mp3":
        audio, sr = librosa.load(file_path)
        librosa.output.write_wav(f"temp_{file_name}", audio, sr)
        file_path = f"temp_{file_name}"
    elif file_ext == ".flac":
        os.system(f"ffmpeg -i {file_path} -acodec pcm_s16le -ar 16000 temp_{file_name}")
        file_path = f"temp_{file_name}"
    elif file_ext == ".mp4":
        clip = VideoFileClip(file_path, audio_codec="aac")
        audio = clip.audio
        audio.write_audiofile(f"temp_{file_name}")
        file_path = f"temp_{file_name}"

    return file_path

def clone(text, url, language):
    """Generate a voice clone using the given parameters."""
    response = requests.get(url)

    with open("temp.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()

    audio_file = [f for f in os.listdir(".") if f.endswith(".wav")][0]

    # Convert the audio file to WAV format
    if os.path.splitext(audio_file)[1].lower() not in [".wav", ".flac"]:
        audio_file = convert_audio_to_wav(audio_file)

    tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path="./output.wav")

    os.remove(audio_file)
    os.remove("temp.zip")

    return "./output.wav"

iface = gr.Interface(fn=clone,
                     inputs=["text", gr.components.Text(label="URL"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label="Language")],
                     outputs=gr.Audio(type='filepath'),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )
                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))

iface.launch(share=True)

    with open("temp.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("temp.zip", "r") as zip_ref:
        zip_ref.extractall()

    audio_file = [f for f in os.listdir(".") if f.endswith(".wav")][0]

    # Convert the audio file to the desired format
    if output_format == "mp3":
        audio, sr = librosa.load(audio_file)
        librosa.output.write_wav("temp.wav", audio, sr)
        os.system(f"ffmpeg -i temp.wav -acodec libmp3lame -qscale:a 4 {audio_file[:-4]}.mp3")
        os.remove("temp.wav")
        audio_file = f"{audio_file[:-4]}.mp3"
    elif output_format == "flac":
        os.system(f"ffmpeg -i {audio_file} {audio_file[:-4]}.flac")
        audio_file = f"{audio_file[:-4]}.flac"
    elif output_format == "mp4":
        clip = VideoFileClip(audio_file, audio_codec="aac")
        clip.write_videofile("output.mp4", fps=24, codec="libx264")
        audio_file = "output.mp4"

    tts.tts_to_file(text=text, speaker_wav=audio_file, language=language, file_path="./output.wav")

    os.remove(audio_file)
    os.remove("temp.zip")

    return "./output.wav"

iface = gr.Interface(fn=clone,
                     inputs=["text", gr.components.Text(label="URL"), gr.Dropdown(choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"], label="Language"), gr.Dropdown(choices=["wav", "mp3", "flac", "mp4"], label="Output Format")],
                     outputs=gr.Audio(type='filepath'),
                     title='Voice Clone',
                     description="""
                     by [Angetyde](https://youtube.com/@Angetyde?si=7nusP31nTumIkPTF) and [Tony Assi](https://www.tonyassi.com/ )
                    use this colab with caution <3.
                     """,
                     theme=gr.themes.Base(primary_hue="teal", secondary_hue="teal", neutral_hue="slate"))

iface.launch(share=True)
