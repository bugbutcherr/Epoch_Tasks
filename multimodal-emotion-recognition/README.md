Multimodal Emotion Recognition (RAVDESS)
This repository contains a complete deep learning pipeline for Multimodal Emotion Recognition using the RAVDESS dataset. It classifies 8 human emotions by processing:

Audio: Converted into Mel-Spectrograms and processed via a Convolutional Neural Network (CNN).
Text: Transcribed from audio using Hugging Face's Whisper model (openai/whisper-tiny), and processed via a Long Short-Term Memory (LSTM) network.
Project Structure
dataset.py: Handles downloading the dataset from Zenodo, audio preprocessing (librosa Mel-spectrogram extraction), and generating text transcripts using Whisper.
models.py: Contains Keras model architectures for Audio CNN, Text LSTM, and an Early Fusion model.
train.py: Training loops, class imbalance handling, Early/Late fusion evaluation, and plot generation.
main.py: The entrypoint to run the entire pipeline end-to-end.
requirements.txt: Python dependencies.
Technical_Report.md: A detailed write-up on architecture decisions, methodology, and results.
Setup & Installation
Clone this repository.
Install dependencies:
pip install -r requirements.txt
Usage
To run the full pipeline (download data, preprocess, train, and evaluate):

python main.py
Options:
--max_samples 100: Only process a subset of audio files (useful for testing).
--epochs 50: Set the number of training epochs.
--skip_preprocessing: Skip the download/whisper transcriptions if you have already generated X_audio.npy and X_text.npy.
Methodology
The system employs both Early Fusion (concatenating bottleneck layers from the Audio and Text networks and training joint dense layers) and Late Fusion (averaging/maxing the softmax outputs of independently trained unimodal networks) to leverage correlations between vocal prosody and spoken content.

See Technical_Report.md for in-depth architectural details and results.
