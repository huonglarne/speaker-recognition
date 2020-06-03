import pydub
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

pydub.AudioSegment.converter = r" c:\users\dell\appdata\local\pip\cache\wheels\64\80\6e\caa3e16deb0267c3cbfd36862058a724144e19fdb9eb03af0f"

path = "D:\pythonprojects\signal\dsp-speaker-recognition\data\Full_train_audio"

def segment_audio(path):
    files = os.listdir(path)
    for name in files:
        file_path = os.path.join(path, name)
        audio = AudioSegment.from_file(file_path, "wav")
        chunk_length_ms = 5000
        chunks = make_chunks(audio, chunk_length_ms)

        for i, chunk in enumerate(chunks):
            print("exporting "+ str(i))
            chunk.export("Seg"+name[4:-4]+"-"+str(i), format="wav")

segment_audio(path)
