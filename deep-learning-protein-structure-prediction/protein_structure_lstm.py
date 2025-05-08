import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import string

# Sample amino acid sequence and structure
sequences = ['ACDEFGHIKLMNPQRSTVWY', 'KLMNPQRSTVWYACDEFGHI']
structures = ['CCCCCHHHHHHHHCCCCCCC', 'HHHHHCCCCCCCHHHHHCCC']  # C: Coil, H: Helix, E: Strand

# Amino acid to one-hot encoding
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_to_int = dict((c, i) for i, c in enumerate(amino_acids))

def one_hot_encode_sequence(seq):
    encoding = np.zeros((len(seq), len(amino_acids)))
    for i, aa in enumerate(seq):
        if aa in aa_to_int:
            encoding[i, aa_to_int[aa]] = 1
    return encoding

X = np.array([one_hot_encode_sequence(seq) for seq in sequences])

# Encode structure labels: C, H, E -> 0, 1, 2
structure_encoder = LabelEncoder()
structure_encoder.fit(['C', 'H', 'E'])
y = [structure_encoder.transform(list(s)) for s in structures]
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build LSTM Model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Predict
predictions = model.predict(X_test)
print("Prediction shape:", predictions.shape)


predicted_classes = np.argmax(predictions, axis=-1)
decoded_preds = [structure_encoder.inverse_transform(p) for p in predicted_classes]

for i, pred in enumerate(decoded_preds):
    print(f"Predicted Structure {i+1}: {''.join(pred)}")