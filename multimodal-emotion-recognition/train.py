import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from models import build_audio_model, build_text_model, build_early_fusion_model, compile_model

# RAVDESS emotion mapping for labels
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def load_data(data_dir):
    """Loads preprocessed data from disk."""
    X_audio = np.load(os.path.join(data_dir, 'X_audio.npy'))
    X_text = np.load(os.path.join(data_dir, 'X_text.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X_audio, X_text, y

def plot_history(history, title, save_path):
    """Plots training and validation accuracy/loss."""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(y_true, y_pred_probs, title, output_dir):
    """Evaluates the model, prints classification report, and plots confusion matrix."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"\n--- {title} Results ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))
    
    plot_confusion_matrix(y_true, y_pred, title, os.path.join(output_dir, f'cm_{title.replace(" ", "_").lower()}.png'))
    return acc

def train_and_evaluate(data_dir="data/processed", output_dir="results", epochs=30, batch_size=32):
    """Main training loop for all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    X_audio, X_text, y = load_data(data_dir)
    print(f"Data Loaded. Audio shape: {X_audio.shape}, Text shape: {X_text.shape}, Labels: {y.shape}")
    
    # 2. Train/Test Split (Use same seed to ensure alignment across modalities)
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=42, stratify=y)
    
    X_audio_train, X_audio_test = X_audio[idx_train], X_audio[idx_test]
    X_text_train, X_text_test = X_text[idx_train], X_text[idx_test]
    
    # Calculate Class Weights to handle imbalance
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_array))
    print(f"Class Weights: {class_weights}")
    
    input_shape_audio = X_audio.shape[1:]
    input_shape_text_len = X_text.shape[1]
    
    # 3. Train Audio-Only Model
    print("\nTraining Audio CNN Model...")
    audio_model, _, _ = build_audio_model(input_shape=input_shape_audio)
    audio_model = compile_model(audio_model)
    history_audio = audio_model.fit(
        X_audio_train, y_train, 
        validation_data=(X_audio_test, y_test), 
        epochs=epochs, batch_size=batch_size, 
        class_weight=class_weights, verbose=1
    )
    plot_history(history_audio, "Audio CNN", os.path.join(output_dir, "history_audio.png"))
    audio_preds = audio_model.predict(X_audio_test)
    acc_audio = evaluate_model(y_test, audio_preds, "Audio CNN", output_dir)
    
    # 4. Train Text-Only Model
    print("\nTraining Text LSTM Model...")
    text_model, _, _ = build_text_model(max_len=input_shape_text_len)
    text_model = compile_model(text_model)
    history_text = text_model.fit(
        X_text_train, y_train, 
        validation_data=(X_text_test, y_test), 
        epochs=epochs, batch_size=batch_size, 
        class_weight=class_weights, verbose=1
    )
    plot_history(history_text, "Text LSTM", os.path.join(output_dir, "history_text.png"))
    text_preds = text_model.predict(X_text_test)
    acc_text = evaluate_model(y_test, text_preds, "Text LSTM", output_dir)
    
    # 5. Train Early Fusion Model
    print("\nTraining Early Fusion Model...")
    fusion_model = build_early_fusion_model(audio_input_shape=input_shape_audio, text_max_len=input_shape_text_len)
    fusion_model = compile_model(fusion_model)
    history_fusion = fusion_model.fit(
        [X_audio_train, X_text_train], y_train, 
        validation_data=([X_audio_test, X_text_test], y_test), 
        epochs=epochs, batch_size=batch_size, 
        class_weight=class_weights, verbose=1
    )
    plot_history(history_fusion, "Early Fusion", os.path.join(output_dir, "history_early_fusion.png"))
    fusion_preds = fusion_model.predict([X_audio_test, X_text_test])
    acc_early_fusion = evaluate_model(y_test, fusion_preds, "Early Fusion", output_dir)
    
    # 6. Evaluate Late Fusion Models (Inference-only combinations)
    print("\nEvaluating Late Fusion Models...")
    
    # Averaging
    late_fusion_avg_preds = (audio_preds + text_preds) / 2.0
    acc_late_avg = evaluate_model(y_test, late_fusion_avg_preds, "Late Fusion (Averaging)", output_dir)
    
    # Weighted Averaging (0.7 Audio + 0.3 Text)
    late_fusion_weighted_preds = (0.7 * audio_preds) + (0.3 * text_preds)
    acc_late_weighted = evaluate_model(y_test, late_fusion_weighted_preds, "Late Fusion (Weighted)", output_dir)
    
    # Max Rule
    # For each sample, take the prediction vector that has the highest max probability
    late_fusion_max_preds = []
    for i in range(len(y_test)):
        if np.max(audio_preds[i]) > np.max(text_preds[i]):
            late_fusion_max_preds.append(audio_preds[i])
        else:
            late_fusion_max_preds.append(text_preds[i])
    late_fusion_max_preds = np.array(late_fusion_max_preds)
    acc_late_max = evaluate_model(y_test, late_fusion_max_preds, "Late Fusion (Max Rule)", output_dir)
    
    # 7. Summary
    print("\n=== FINAL ACCURACY SUMMARY ===")
    print(f"Audio CNN:               {acc_audio:.4f}")
    print(f"Text LSTM:               {acc_text:.4f}")
    print(f"Early Fusion:            {acc_early_fusion:.4f}")
    print(f"Late Fusion (Avg):       {acc_late_avg:.4f}")
    print(f"Late Fusion (Weighted):  {acc_late_weighted:.4f}")
    print(f"Late Fusion (Max):       {acc_late_max:.4f}")

if __name__ == "__main__":
    # Test block
    train_and_evaluate(epochs=2, batch_size=8)
