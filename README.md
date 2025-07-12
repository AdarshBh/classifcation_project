# classifcation_project

# 🧠 Mental Health Text Classification using BiLSTM with Data Augmentation

This project aims to classify mental health statements into various mental health conditions such as **Depression, Anxiety, Bipolar, Stress, Personality Disorder**, etc., using a deep learning model with a BiLSTM architecture. It also tackles class imbalance using advanced **EDA-based textual data augmentation**.

---

## 📁 Dataset

- **File**: `Combined Data.csv`
- **Columns**:
  - `statement`: The textual input.
  - `status`: The label indicating the mental health condition.

---

## 📌 Key Features

- 📊 Exploratory Data Analysis (EDA) with visualizations.
- 🧠 Class imbalance handled with EDA-based data augmentation (synonym replacement, random insertion, swap, deletion).
- 🔤 Text preprocessing using tokenization, stopword removal, and lemmatization.
- 🤖 Deep Learning with **Bidirectional LSTM** for multi-class classification.
- 🧮 Class weighting to combat residual imbalance.
- 📉 Early stopping and model checkpointing to prevent overfitting.

---

## 🛠️ Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn` – for data manipulation and visualization  
- `nltk` – for text processing and synonym augmentation  
- `tensorflow` / `keras` – for building and training the deep learning model  
- `sklearn` – for preprocessing, metrics, and splitting  
- `tqdm` – for progress monitoring  
- `transformers` – for potential future integration with LLMs  

---

## 🧪 Data Augmentation Techniques

Implemented via **Easy Data Augmentation (EDA)**:

- ✅ Synonym Replacement  
- ✅ Random Insertion  
- ✅ Random Swap  
- ✅ Random Deletion  

Used to augment underrepresented classes to reach a target size of 7000 samples per class.

---

## 🏗️ Model Architecture

- **Embedding Layer**: Word embeddings learned during training.  
- **Bidirectional LSTM (128)** → Dropout  
- **Bidirectional LSTM (64)** → Dropout  
- **Dense Layer (64)** → Dropout  
- **Output Layer**: Softmax over number of classes  

---

## 📈 Training Details

- **Loss Function**: Sparse Categorical Crossentropy  
- **Optimizer**: Adam  
- **Epochs**: 10 (with early stopping)  
- **Batch Size**: 64  
- **Max Words**: 6000  
- **Max Sequence Length**: 128  
- **Callbacks**:
  - EarlyStopping (patience=3)
  - ModelCheckpoint (`best_lstm_model.keras`)

---

## 📊 Evaluation Metrics

- ✅ Accuracy  
- ✅ Classification Report (Precision, Recall, F1-score)  
- ✅ Confusion Matrix (saved as `confusion_matrix.png`)  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
##then execute
python mental_health_classification.py
