from bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
from hubert.hubert_manager import HuBERTManager
from hubert.pre_kmeans_hubert import CustomHubert
from hubert.customtokenizer import CustomTokenizer
from bark.api import generate_audio
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

import torchaudio
import torch
from scipy.io.wavfile import write as write_wav

device = 'cuda' # or 'cpu'
model = load_codec_model(use_gpu=True if device == 'cuda' else False)

# hubert_manager = HuBERTManager()
# hubert_manager.make_sure_hubert_installed()
# hubert_manager.make_sure_tokenizer_installed()

# Load the HuBERT model
hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').to(device)

# Load the CustomTokenizer model
tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').to(device)  # Automatically uses the right layers

audio_filepath = './samples_ref/ref_KSS_02.wav' # the audio you want to clone (under 13 seconds)
wav, sr = torchaudio.load(audio_filepath)
wav = convert_audio(wav, sr, model.sample_rate, model.channels)
wav = wav.to(device)

semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
semantic_tokens = tokenizer.get_token(semantic_vectors)


# Extract discrete codes from EnCodec
with torch.no_grad():
    encoded_frames = model.encode(wav.unsqueeze(0))
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]

# move codes to cpu
codes = codes.cpu().numpy()
# move semantic tokens to cpu
semantic_tokens = semantic_tokens.cpu().numpy()

import numpy as np
voice_name = 'clone_actor01_01' # whatever you want the name of the voice to be
output_path = 'bark/assets/prompts/' + voice_name + '.npz'
np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)

# Enter your prompt and speaker here
text_prompt = "안녕하세요. 한국전자기술연구원 지능형영상처리센터입니다."
# "Hello. This is the Intelligent Image Processing Center at the Korea Electronics Technology Institute."
voice_name = "clone_actor01_01" # use your custom voice name here if you have one

# download and load all models
preload_models(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
    path="models"
)
# history_prompt=voice_name,
# simple generation
audio_array = generate_audio(text_prompt, history_prompt="ko_speaker_1", text_temp=0.7, waveform_temp=0.7)

# save audio
filepath = "./samples_cloned/bark_ko_speaker_01.wav" # change this to your desired output path
write_wav(filepath, SAMPLE_RATE, audio_array)