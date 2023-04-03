'''
This script runs the ESPnet TTS inference. 
'''

# The following packages are required to be install at first
#pip install espnet==0.10.6 pyopenjtalk==0.2 pypinyin==0.44.0 parallel_wavegan==0.5.4 gdown==4.4.0 espnet_model_zoo

from espnet2.bin.tts_inference import Text2Speech
import scipy.io.wavfile
import argparse
import pandas as pd

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('query', type=str,
                    help='query to be searched on the reference corpus')

args = parser.parse_args()

# decide the input sentence by yourself
#print(f"Input your favorite sentence in {lang}.")
x = "hello gronings"

tts_hoog = Text2Speech.from_pretrained(
  model_tag="https://huggingface.co/ahnafsamin/FastSpeech2-gronings/resolve/main/tts_train_fastspeech2_raw_char_tacotron_train.loss.ave.zip",
  vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v3",
  speed_control_alpha=1.0
)

tts_west = Text2Speech.from_pretrained(
  model_tag="https://huggingface.co/ahnafsamin/FastSpeech2-gronings-westerkwartiers/resolve/main/tts_train_fastspeech2_raw_char_tacotron_train.loss.ave.zip",
  vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v3",
  speed_control_alpha=1.0
)

tts_old = Text2Speech.from_pretrained(
  model_tag="https://huggingface.co/ahnafsamin/FastSpeech2-gronings-oldambster/resolve/main/tts_train_fastspeech2_raw_char_tacotron_train.loss.ave.zip",
  vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v3",
  speed_control_alpha=1.0
)

# synthesis
speech_hoog = tts_hoog(args.query)["wav"]
speech_west = tts_west(args.query)["wav"]
speech_old = tts_old(args.query)["wav"]

scipy.io.wavfile.write("HO_" + args.query + ".wav", tts_hoog.fs , speech_hoog.view(-1).cpu().numpy())
scipy.io.wavfile.write("WE_" + args.query + ".wav", tts_west.fs , speech_west.view(-1).cpu().numpy())
scipy.io.wavfile.write("OL_" + args.query + ".wav", tts_old.fs , speech_old.view(-1).cpu().numpy())
