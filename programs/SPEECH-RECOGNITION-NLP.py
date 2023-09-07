import os
import random

import librosa
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")

path = "../datasets/4-SPEECH-RECOGNITION/"

audio_file = os.path.join(path, random.choice(os.listdir(path)))
audio, sr = librosa.load(audio_file, sr=16000)

input_features = processor(audio, return_tensors="tf", sampling_rate=16000).input_features
generated_ids = model.generate(input_features=input_features, max_new_tokens=420)
transcription = processor.batch_decode(generated_ids, )[0]
print(transcription)
