import argparse
import os
from dataset import preprocess_data
from train import train_and_evaluate

def main():
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition on RAVDESS")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory to download/extract RAVDESS dataset.")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory to save preprocessed data.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results and plots.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max audio files to process (useful for testing).")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing and just train on existing data.")
    
    args = parser.parse_args()
    
    if not args.skip_preprocessing:
        print("=== Step 1: Data Preprocessing ===")
        preprocess_data(args.data_dir, args.processed_dir, max_samples=args.max_samples)
    else:
        print("=== Step 1: Skipping Data Preprocessing ===")
        
    print("\n=== Step 2: Training and Evaluation ===")
    if not os.path.exists(os.path.join(args.processed_dir, 'X_audio.npy')):
        print("Error: Preprocessed data not found. Please run without --skip_preprocessing first.")
        return
        
    train_and_evaluate(data_dir=args.processed_dir, output_dir=args.results_dir, epochs=args.epochs, batch_size=args.batch_size)
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
