import os
import glob
import numpy as np
import librosa
from transformers import pipeline
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import urllib.request
import zipfile

# RAVDESS emotion mapping (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
EMOTION_DICT = {
    '01': 0, # neutral
    '02': 1, # calm
    '03': 2, # happy
    '04': 3, # sad
    '05': 4, # angry
    '06': 5, # fearful
    '07': 6, # disgust
    '08': 7  # surprised
}

def download_and_extract(data_dir):
    """Downloads RAVDESS sample or full dataset if not exists."""
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'Audio_Speech_Actors_01-24.zip')
    
    # We provide the exact Zenodo link for RAVDESS audio
    # Due to size, we might skip downloading if files exist
    if not os.path.exists(zip_path) and len(glob.glob(os.path.join(data_dir, 'Actor_*'))) == 0:
        print("Downloading RAVDESS dataset from Zenodo (This may take a while - ~400MB)...")
        url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, max_len=130):
    """Loads audio and extracts Mel-spectrogram."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        # Apply trimming to remove silence
        y, _ = librosa.effects.trim(y)
        
        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Pad or truncate to max_len
        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_len]
            
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_data(data_dir, output_dir, max_samples=None):
    """Processes audio files to Mel-spectrograms and generates text transcripts."""
    download_and_extract(data_dir)
    
    file_paths = glob.glob(os.path.join(data_dir, 'Actor_*', '*.wav'))
    if max_samples:
        file_paths = file_paths[:max_samples]
        
    print(f"Found {len(file_paths)} audio files. Initializing Whisper for speech-to-text...")
    # Initialize whisper ASR pipeline
    asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    
    audio_features = []
    text_transcripts = []
    labels = []
    
    for file_path in tqdm(file_paths, desc="Processing files"):
        # Extract label from filename
        # Format: 03-01-01-01-01-01-01.wav -> Emotion is 3rd element
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        emotion_code = parts[2]
        if emotion_code not in EMOTION_DICT:
            continue
            
        label = EMOTION_DICT[emotion_code]
        
        # 1. Process Audio
        mel_spec = extract_mel_spectrogram(file_path)
        if mel_spec is None:
            continue
            
        # 2. Process Text (Generate Transcript)
        transcript = ""
        try:
            result = asr_pipeline(file_path)
            transcript = result['text'].strip()
        except Exception as e:
            print(f"ASR failed for {file_path}: {e}")
            transcript = ""
            
        audio_features.append(mel_spec)
        text_transcripts.append(transcript)
        labels.append(label)
        
    # Tokenize Text
    print("Tokenizing transcripts...")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(text_transcripts)
    text_sequences = tokenizer.texts_to_sequences(text_transcripts)
    
    # Pad sequences
    max_text_len = 50 # Usually RAVDESS sentences are short ("Kids are talking by the door")
    text_padded = pad_sequences(text_sequences, maxlen=max_text_len, padding='post', truncating='post')
    
    # Convert to numpy arrays
    X_audio = np.array(audio_features)
    # Add channel dimension for CNN
    X_audio = np.expand_dims(X_audio, axis=-1)
    
    X_text = np.array(text_padded)
    y = np.array(labels)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_audio.npy'), X_audio)
    np.save(os.path.join(output_dir, 'X_text.npy'), X_text)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    
    import pickle
    with open(os.path.join(output_dir, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print(f"Saved preprocessed data to {output_dir}")
    return X_audio, X_text, y, tokenizer

if __name__ == "__main__":
    # Test block
    preprocess_data("data/raw", "data/processed", max_samples=10)
