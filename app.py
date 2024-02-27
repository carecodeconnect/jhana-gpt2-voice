# import streamlit as st
# import numpy as np
# import whisper
# import transformers
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from TTS.api import TTS
# import torch
# import os
# import io
# transformers.logging.set_verbosity_info()

# # Assume the base directory is where the script is located, no need to change directory
# # os.chdir(os.path.dirname(os.path.abspath(__file__)))

# # Adjust the path to be relative to the current file, ensuring compatibility with different environments
# base_dir = os.path.dirname(os.path.abspath(__file__))
# speaker_wav_path = os.path.join(base_dir, "data/input/voices_to_clone/audio_cf_10_seconds.wav")
# if not os.path.exists(speaker_wav_path):
#     raise FileNotFoundError(f"The specified speaker WAV file does not exist: {speaker_wav_path}")

# #model_name = "carecodeconnect/jhana-gpt2"
# model_name = "gpt2"

# def process_audio_file(uploaded_file):
#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()
#         audio_data = np.frombuffer(bytes_data, dtype=np.int16)
#         return audio_data
#     else:
#         return None

# def speech_to_text(audio_data):
#     if audio_data is not None:
#         model = whisper.load_model("small")
#         result = model.transcribe(audio_data)
#         return result["text"]
#     else:
#         return ""

# def generate_text_with_gpt2(input_text):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     generated_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.5)
#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# def text_to_speech(input_text, output_dir="data/output/audio/", tts_model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=speaker_wav_path):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     output_dir = os.path.join(base_dir, output_dir)  # Ensure the output directory is correctly set relative to the base directory
#     os.makedirs(output_dir, exist_ok=True)
#     output_file_path = os.path.join(output_dir, "response_speech.wav")

#     try:
#         # This section might need adjustment based on how the TTS API handles non-interactive environments in Hugging Face Spaces
#         tts = TTS(tts_model_path).to(device)
#         tts.tts_to_file(text=input_text, file_path=output_file_path, language="en", speaker_wav=speaker_wav_path)
#         return output_file_path
#     except Exception as e:
#         st.error(f"Error using TTS model: {e}")
#         return None

# def main():
#     st.title("Voice-Interactive GPT-2 App with Voice Cloning")

#     interaction_mode = st.radio("Choose your interaction mode:", ("Text", "Upload Audio"))

#     input_text = ""
#     if interaction_mode == "Text":
#         input_text = st.text_area("Enter your text here:")
#         process_button = st.button("Process Text")
#     else:
#         uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
#         if uploaded_file is not None:
#             audio_data = process_audio_file(uploaded_file)
#             input_text = speech_to_text(audio_data)
#             st.write("Transcribed Text:", input_text)
#             process_button = True
#         else:
#             process_button = False

#     if process_button and input_text:
#         generated_text = generate_text_with_gpt2(input_text)
#         st.write("Generated Text:", generated_text)

#         response_audio_path = text_to_speech(generated_text)
#         if response_audio_path:
#             with open(response_audio_path, 'rb') as audio_file:
#                 audio_bytes = audio_file.read()
#             st.audio(audio_bytes, format='audio/wav')

# if __name__ == "__main__":
#     main()
# import streamlit as st
# import numpy as np
# import whisper
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch
# import os
# import tempfile
# from TTS.utils.manage import ModelManager
# from TTS.utils.synthesizer import Synthesizer

# # Streamlit logging adjustment
# st.set_option('deprecation.showfileUploaderEncoding', False)

# # Initialize Model Manager for TTS
# model_manager = ModelManager()

# # Assume the base directory is where the script is located, no need to change directory
# base_dir = os.path.dirname(os.path.abspath(__file__))
# speaker_wav_path = os.path.join(base_dir, "data/input/voices_to_clone/audio_cf_10_seconds.wav")
# if not os.path.exists(speaker_wav_path):
#     raise FileNotFoundError(f"The specified speaker WAV file does not exist: {speaker_wav_path}")

# model_name = "gpt2"

# def process_audio_file(uploaded_file):
#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()
#         audio_data = np.frombuffer(bytes_data, dtype=np.int16)
#         return audio_data
#     else:
#         return None

# def speech_to_text(audio_data):
#     if audio_data is not None:
#         model = whisper.load_model("small")
#         result = model.transcribe(audio_data)
#         return result["text"]
#     else:
#         return ""

# def generate_text_with_gpt2(input_text):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
#     input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     generated_ids = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.5)
#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# def text_to_speech(input_text, model_name="en/ljspeech/tacotron2-DDC"):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     output_dir = os.path.join(base_dir, "data/output/audio/")
#     os.makedirs(output_dir, exist_ok=True)
#     output_file_path = os.path.join(output_dir, "response_speech.wav")

#     try:
#         model_path, config_path, _ = model_manager.download_model(f"tts_models/{model_name}")
#         synthesizer = Synthesizer(model_path, config_path)
#         wavs = synthesizer.tts(input_text)
#         with open(output_file_path, 'wb') as f_out:
#             synthesizer.save_wav(wavs, f_out)
#         return output_file_path
#     except Exception as e:
#         st.error(f"Error using TTS model: {e}")
#         return None

# def text_to_speech(input_text, output_dir="data/output/audio/", model_name="multilingual/multi-dataset/xtts_v2", speaker_wav=None):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     output_dir = os.path.join(base_dir, output_dir)  # Ensure the output directory is correctly set relative to the base directory
#     os.makedirs(output_dir, exist_ok=True)
#     output_file_path = os.path.join(output_dir, "response_speech.wav")

#     # Check if speaker_wav is provided, if not, use the default path
#     if speaker_wav is None:
#         speaker_wav = speaker_wav_path
#     # Ensure the speaker_wav file exists
#     if not os.path.exists(speaker_wav):
#         st.error(f"Speaker WAV file does not exist: {speaker_wav}")
#         return None

#     try:
#         # Download and prepare the TTS model and its configuration
#         model_path, config_path, _ = model_manager.download_model(f"tts_models/{model_name}")
#         if model_path is None or config_path is None:
#             st.error("Failed to download TTS model. Please check the model name and internet connection.")
#             return None

#         synthesizer = Synthesizer(model_path, config_path, use_cuda=device=='cuda')

#         # Generate speech using the TTS model, specifying the language and the speaker wav for voice cloning
#         wavs = synthesizer.tts(input_text, language="en", speaker_wav=speaker_wav)

#         # Save the generated speech to a WAV file
#         with open(output_file_path, 'wb') as f_out:
#             synthesizer.save_wav(wavs, f_out)

#         return output_file_path
#     except Exception as e:
#         st.error(f"Error using TTS model: {e}")
#         return None

    
# def main():
#     st.title("Voice-Interactive GPT-2 App with Voice Cloning")

#     interaction_mode = st.radio("Choose your interaction mode:", ("Text", "Upload Audio"))

#     input_text = ""
#     if interaction_mode == "Text":
#         input_text = st.text_area("Enter your text here:")
#         process_button = st.button("Process Text")
#     else:
#         uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
#         if uploaded_file is not None:
#             audio_data = process_audio_file(uploaded_file)
#             input_text = speech_to_text(audio_data)
#             st.write("Transcribed Text:", input_text)
#             process_button = True
#         else:
#             process_button = False

#     if process_button and input_text:
#         generated_text = generate_text_with_gpt2(input_text)
#         st.write("Generated Text:", generated_text)

#         response_audio_path = text_to_speech(generated_text)
#         if response_audio_path:
#             st.audio(response_audio_path, format='audio/wav')

# if __name__ == "__main__":
#     main()

import streamlit as st
import numpy as np
import whisper
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS

# Initialize Model Manager for TTS
model_manager = ModelManager()

base_dir = os.path.dirname(os.path.abspath(__file__))
speaker_wav_path = os.path.join(base_dir, "data/input/voices_to_clone/audio_cf_10_seconds.wav")
if not os.path.exists(speaker_wav_path):
    raise FileNotFoundError(f"The specified speaker WAV file does not exist: {speaker_wav_path}")

model_name = "gpt2"

def process_audio_file(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        audio_data = np.frombuffer(bytes_data, dtype=np.int16)
        return audio_data
    else:
        return None

def speech_to_text(audio_data):
    if audio_data is not None:
        model = whisper.load_model("small")
        result = model.transcribe(audio_data)
        return result["text"]
    else:
        return ""
    
def generate_text_with_gpt2(input_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Explicitly setting the attention mask
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)  # Assuming all tokens are to be attended to
    generated_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=100, do_sample=True, temperature=0.5)
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def text_to_speech(input_text, output_dir="data/output/audio/", tts_model_name="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=speaker_wav_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Resolve paths relative to the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    tts_model_path = os.path.join(script_dir, tts_model_name)
    speaker_wav_path = os.path.join(script_dir, speaker_wav_path)

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "response_speech.wav")

    try:
        tts = TTS(tts_model_path).to(device)
        tts.tts_to_file(text=input_text, file_path=output_file_path, language="en", speaker_wav=speaker_wav_path)
        return output_file_path
    except Exception as e:
        st.error(f"Error using TTS model: {e}")
        return None

def main():
    st.title("Voice-Interactive GPT-2 App with Voice Cloning")

    interaction_mode = st.radio("Choose your interaction mode:", ("Text", "Upload Audio"))

    input_text = ""
    if interaction_mode == "Text":
        input_text = st.text_area("Enter your text here:")
        process_button = st.button("Process Text")
    else:
        uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
        if uploaded_file is not None:
            audio_data = process_audio_file(uploaded_file)
            input_text = speech_to_text(audio_data)
            st.write("Transcribed Text:", input_text)
            process_button = True
        else:
            process_button = False

    if process_button and input_text:
        generated_text = generate_text_with_gpt2(input_text)
        st.write("Generated Text:", generated_text)

        response_audio_path = text_to_speech(generated_text)
        if response_audio_path:
            st.audio(response_audio_path, format='audio/wav')

if __name__ == "__main__":
    main()