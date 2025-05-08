import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define path to your CASP7 dataset (update the path)
casp7_data_dir = "path_to_casp7_data"  # Change this to the correct path to your CASP7 data folder

# Function to load FASTA sequences and secondary structure labels from CASP7
def load_casp7_data(data_dir):
    sequences = []
    structures = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".fasta"):
            # Load protein sequences from FASTA files
            with open(os.path.join(data_dir, file_name), "r") as f:
                lines = f.readlines()
                seq = "".join([line.strip() for line in lines if not line.startswith(">")])
                sequences.append(seq)
                
                # Corresponding secondary structure file (e.g., .ss)
                ss_file = file_name.replace(".fasta", ".ss")
                with open(os.path.join(data_dir, ss_file), "r") as f_ss:
                    ss = f_ss.readlines()[0].strip()
                    structures.append(ss)
    
    return sequences, structures

# Function to convert sequences and structures to suitable format
def preprocess_data(sequences, structures, max_len=1000):
    # Label encode the secondary structure labels (H, E, C)
    label_encoder = LabelEncoder()
    structures = label_encoder.fit_transform([s if s == "H" else "E" if s == "E" else "C" for s in structures])
    
    # Map amino acids to integers for one-hot encoding
    aa_dict = { 'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'G': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
                'T': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  
                'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}  
    
    def one_hot(seq):
        return np.array([aa_dict.get(aa, aa_dict['X']) for aa in seq])
    
    # One-hot encode all sequences
    X = np.array([one_hot(seq) for seq in sequences])
    
    # Pad sequences to the same length
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post')
    
    # Pad/adjust structure labels (one-hot encoding)
    y = np.array([np.pad(struct, (0, max_len - len(struct)), 'constant', constant_values=-1) for struct in structures])
    
    return X, y

# Load and preprocess data
sequences, structures = load_casp7_data(casp7_data_dir)
X, y = preprocess_data(sequences, structures)

# Define Transformer Model for secondary structure prediction
def transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    # Transformer Encoder Layer
    x = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
    x = Dropout(0.2)(x)
    x = LayerNormalization()(x)
    
    # Fully connected output layer
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = LayerNormalization()(x)
    
    # Final classification layer
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model

# Define the input shape based on your preprocessed data
input_shape = (X.shape[1], X.shape[2])  # Sequence length x feature size
num_classes = 3  # H, E, C

# Initialize and compile the model
model = transformer_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('protein_structure_transformer.h5')

# Example prediction on a new protein sequence
test_sequence = 'AGCTAGCTAGC'  # Example test sequence
test_sequence_onehot = np.array([one_hot(test_sequence)])
test_sequence_onehot = pad_sequences(test_sequence_onehot, maxlen=X.shape[1], padding='post')
prediction = model.predict(test_sequence_onehot)
print(f"Predicted Secondary Structure: {prediction}")
