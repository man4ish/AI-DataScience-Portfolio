# Protein Secondary Structure Prediction using LSTM

This project demonstrates how to use a simple deep learning model (LSTM) to predict the secondary structure (Helix, Coil, Strand) of protein sequences from amino acid sequences using Python and TensorFlow.

---

## Project Structure

```
protein-secondary-structure-prediction/
│
├── protein_structure_lstm.py       # Main script with LSTM model
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview
```

---

## Description

Protein secondary structure prediction is a key problem in computational biology. This project uses a Long Short-Term Memory (LSTM) neural network to classify each amino acid in a protein sequence into one of three secondary structure types:

- **H**: Alpha-Helix  
- **E**: Beta-Strand  
- **C**: Coil

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/protein-secondary-structure-prediction.git
cd protein-secondary-structure-prediction
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv env
source env/bin/activate  # or use env\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Model

```bash
python protein_structure_lstm.py
```

---

## Output

- Trains on sample protein sequences and structure labels.
- Prints model accuracy per epoch.
- Outputs predicted secondary structure for each amino acid.

Example:
```
Predicted Structure 1: CCHHHHHCCCCCHHHHCCC
```

---

## Requirements

See `requirements.txt`:

```txt
tensorflow==2.13.0
numpy
scikit-learn
```

---

## Notes

- Uses dummy sequences (`ACDEFG...`) and labels (`CHHHHCC...`) for illustration.
- You can extend this model to use real datasets like **CB513** or **ProteinNet**.
- Future versions may include Transformer-based architectures like ProtBERT.

---

## Author

Manish Kumar

---