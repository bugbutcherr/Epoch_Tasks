import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Embedding, LSTM, GRU, Concatenate
from tensorflow.keras.optimizers import Adam

def build_audio_model(input_shape=(128, 130, 1), num_classes=8):
    """Builds a CNN model for audio classification."""
    audio_input = Input(shape=input_shape, name="audio_input")
    
    # Conv Block 1
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(audio_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Conv Block 2
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Conv Block 3
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    
    # Dense layers
    bottleneck = Dense(256, activation='relu', name="audio_bottleneck")(x)
    bottleneck = Dropout(0.4)(bottleneck)
    
    output = Dense(num_classes, activation='softmax', name="audio_output")(bottleneck)
    
    model = Model(inputs=audio_input, outputs=output, name="Audio_CNN")
    return model, audio_input, bottleneck

def build_text_model(vocab_size=5000, max_len=50, num_classes=8):
    """Builds an RNN/LSTM model for text classification."""
    text_input = Input(shape=(max_len,), name="text_input")
    
    # Embedding layer
    x = Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len)(text_input)
    
    # LSTM layer
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    bottleneck = Dense(128, activation='relu', name="text_bottleneck")(x)
    bottleneck = Dropout(0.3)(bottleneck)
    
    output = Dense(num_classes, activation='softmax', name="text_output")(bottleneck)
    
    model = Model(inputs=text_input, outputs=output, name="Text_LSTM")
    return model, text_input, bottleneck

def build_early_fusion_model(audio_input_shape=(128, 130, 1), text_vocab_size=5000, text_max_len=50, num_classes=8):
    """Builds an Early Fusion model combining Audio and Text bottleneck vectors."""
    # Get base models (without the final softmax output)
    _, audio_input, audio_bottleneck = build_audio_model(audio_input_shape, num_classes)
    _, text_input, text_bottleneck = build_text_model(text_vocab_size, text_max_len, num_classes)
    
    # Concatenate bottlenecks
    concat = Concatenate()([audio_bottleneck, text_bottleneck])
    
    # Joint Dense Layers
    x = Dense(256, activation='relu')(concat)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Final Output
    output = Dense(num_classes, activation='softmax', name="fusion_output")(x)
    
    model = Model(inputs=[audio_input, text_input], outputs=output, name="Early_Fusion_Model")
    return model

def compile_model(model, learning_rate=0.001):
    """Compiles the given model with Categorical Cross-Entropy and Adam."""
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    audio_model, _, _ = build_audio_model()
    audio_model.summary()
    
    text_model, _, _ = build_text_model()
    text_model.summary()
    
    fusion_model = build_early_fusion_model()
    fusion_model.summary()
