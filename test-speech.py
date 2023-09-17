from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Sorry no idea how I missed this message. I think I now have too many servers on here drowning out my notifications. Microsoft have a library that seems to have quite a bit of an overlap in functionality with langchain called semantic kernel. It also support .net etc, But to be honest a lot of companies are either just using langchain or creating their own prompts. as its essentially a wrapper aroung different functionalities you might want to have access to yourself if you are building something of enterprise grade. Cause Langchain prompts chains might not suit your specific usecase.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=19000)