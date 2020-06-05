import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

pydub.AudioSegment.converter = r" c:\users\dell\appdata\local\pip\cache\wheels\64\80\6e\caa3e16deb0267c3cbfd36862058a724144e19fdb9eb03af0f"
path = r"D:\pythonprojects\signal\dsp-speaker-recognition\vad_processed"

def segment_audio(file_path):
    audio = AudioSegment.from_file(file_path, "wav")
    chunk_length_ms = 5000
    chunks = make_chunks(audio, chunk_length_ms)

    for i, chunk in enumerate(chunks):
        print(type(chunk))
        name = os.path.basename(file_path)[7:-4]+"-"+str(i)
        print("exporting " + name)
        chunk.export(name, format="wav")

audio_paths = [os.path.join(path, name) for name in os.listdir(path)]
for path in audio_paths[1:]:
    segment_audio(path)
# segment_audio(r"D:\pythonprojects\signal\dsp-speaker-recognition\vad_processed\Retake-Speaker-3.wav")
