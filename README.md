# Deception Detection in Diplomacy Dialogues using a Hybrid Deep Learning Model

This project focuses on the challenging Natural Language Processing (NLP) task of automatically detecting deceptive messages within game dialogues from the strategic board game Diplomacy. The goal is to develop and evaluate a deep learning model capable of classifying messages as either truthful or deceptive ('Lie').

## Abstract

The strategic board game Diplomacy is characterized by complex negotiations and inherent deception. This project details the development and evaluation of a hybrid deep learning model designed for automatically detecting deceptive messages within game dialogues. This is a challenging NLP task with implications for understanding strategic communication and developing sophisticated AI agents. The model combines textual information extracted from messages using a Transformer-based encoder initialized with FastText embeddings, with contextual game state information represented as a graph processed by Graph Convolutional Networks (GCNs). The dataset is derived from Diplomacy game logs, and involves text cleaning, tokenization, and feature engineering, including score differentials and player interaction graphs. Due to significant class imbalance (lies vs. truths), downsampling was applied to the training data. The model architecture integrates text and graph features through concatenation followed by a self-attention mechanism before final classification. Evaluation on a held-out test set shows an overall accuracy of 85.81%, and notably improves upon baseline results for the minority 'Lie' class, achieving a Lie F1-score of 0.2810 and a Macro F1-score of 0.6011. This suggests that while the hybrid approach captures relevant signals and represents progress on this difficult task, further improvements are needed for highly reliable deception detection in this complex domain.

## Introduction

The game of Diplomacy, set in pre-World War I Europe, emphasizes negotiation, alliances, and betrayal. Deception is a core element of successful strategy. Detecting deceptive communication in this context is a complex challenge for AI and NLP. Automated deception detection has broad applications, such as identifying fake news and online scams. In Diplomacy, it can help analyze strategies and build more human-like AI opponents. This work proposes a hybrid approach leveraging:
1.  **Textual Features**: Analyzing message content using a Transformer-based encoder with pre-trained FastText embeddings.
2.  **Contextual Features**: Incorporating game state information (player scores, communication patterns) modeled as a graph processed by GCNs.

## Methodology

### Dataset
The data originates from logs of multiple Diplomacy games in JSON Lines (`.jsonl`) format. Each line typically represents a game segment, containing messages, speakers, recipients, timestamps (year/season), game scores, score changes (deltas), and binary deception labels ('Lie'=0, 'Truth'=1).
The dataset was split as follows:
* Training Set: 13,132 labeled messages.
* Validation Set: 1,416 labeled messages.
* Test Set: 2,741 labeled messages.
A total of 17,289 labeled messages across 12 unique games were available.

### Data Preprocessing
1.  **Text Cleaning**: Lowercasing, removing URLs/mentions/hashtags, removing most punctuation (keeping apostrophes), removing digits, and normalizing whitespace.
2.  **Tokenization**: Cleaned text was tokenized using Keras' `Tokenizer` with a max vocabulary size of 10,000 and an OOV token ('<OOV>'). The actual vocabulary size used was 9,946. Sequences were padded/truncated to a length of 60.
3.  **Class Imbalance Handling**: Downsampling was applied to the training data to address the imbalance (591 'Lie' vs. 12,541 'Truth' in the original training set). After downsampling, the training set had 591 'Lie' and 591 'Truth' samples.

### Feature Engineering
1.  **Textual Embeddings (FastText)**: Pre-trained 300-dimension FastText embeddings (`crawl-300d-2M.vec`) initialized the embedding layer.
2.  **Delta Features**:
    * Current Delta: Speaker score - Recipient score in the current turn.
    * Future Delta: Speaker score - Recipient score in the next turn.
    Missing scores resulted in a delta of 0.0.
3.  **Graph Features (Game State Graph)**: For each turn (game ID, year, season), a graph was built with up to 7 players as nodes.
    * Node Features (1D): Player's game score (0.0 if missing).
    * Adjacency Matrix (7x7): Unweighted, undirected, indicating communication between players in that turn. Self-loops were added ($A' = A + I$).
    * Node Mask (7): Boolean mask for active players.
    312 unique turn graphs were built.

### Model Architecture
A hybrid model (`DiplomacyHybridModel`) was developed, comprising a Text Feature Encoder, a Graph Encoder, and a Fusion/Classification Head.

1.  **Text Feature Encoder**:
    * **Inputs**: Padded token sequences (length 60), Delta features (dim 2).
    * **Layers**: Embedding (300D, FastText initialized, trainable), Multi-Head Attention (4 heads) + Add&Norm, Global Average Pooling, Dropout (0.15), Dense layer for delta features (16D, ReLU), Concatenation, Dropout (0.225), Output Dense (300D, ReLU).
2.  **Graph Encoder**:
    * **Inputs**: Node Features (7x1), Adjacency Matrix (7x7), Node Mask (7).
    * **Layers**: Node Masking, 3x GCN Layers (Units: 64, 64, 32; ReLU; Dropout 0.15 after each), Masked Global Average Pooling, Dropout (0.225), Output Dense (150D, ReLU).
3.  **Fusion and Classification Head**:
    * **Inputs**: Text embedding (300D), Graph embedding (150D).
    * **Layers**: Concatenation (450D), Dense (450D, ReLU), Dropout (0.4), Reshape + Multi-Head Self-Attention (4 heads), Reshape, Dropout (0.4), Output Dense (1 unit, Sigmoid).

## Notebooks Overview

This repository contains several Jupyter Notebooks detailing different experimental rounds and model versions:

* **`deception-detection-group-88.ipynb`**: **This is the main notebook where the primary hybrid model, including the GCN implementation, is detailed.** It incorporates FastText embeddings, GCN for graph-based contextual features, and attention mechanisms. This version achieves the reported project results.
* `deception-f.ipynb`: Contains initial data exploration, quadrant analysis of messages, and an implementation of a Bi-LSTM with Attention model.
* `decpetion-round-2.ipynb`: Focuses on a Bi-LSTM model with attention, explores data balancing techniques, and evaluates based on sender and receiver labels.
* `deception-round-3.ipynb`: Implements a simple Transformer model combined with action features and delta scores, utilizing downsampling for class imbalance.
* `deception-round-4.ipynb`: Details a modular hybrid GCN and Transformer model with attention, using FastText embeddings and downsampling. This notebook appears to be an iteration leading to or similar to `deception-detection-group-88.ipynb`.

## Setup and How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Environment Setup:**
    * Ensure you have Python (e.g., 3.10+) and Jupyter Notebook/JupyterLab installed.
    * It's recommended to use a virtual environment:
        ```bash
        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```
3.  **Install Dependencies:**
    The primary dependencies can be inferred from the notebooks' import statements. Install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow requests tqdm
    # Add any other specific libraries you used
    ```
    * `pandas`, `numpy`: For data manipulation.
    * `tensorflow`, `keras`: For building and training the deep learning model.
    * `scikit-learn`: For metrics, resampling, etc.
    * `matplotlib`, `seaborn`: For plotting.
    * `requests`, `zipfile`, `gzip`, `shutil`: Used in `deception-detection-group-88.ipynb` for downloading and extracting FastText embeddings.

4.  **Dataset:**
    * The notebooks expect the dataset (`train.jsonl`, `validation.jsonl`, `test.jsonl`) to be in a directory specified by the `BASE_DATA_DIR` variable, which defaults to `./Dataset/` or `/kaggle/input/diplomacy/transformers/default/1/Dataset/` if running in a Kaggle environment.
    * Place your dataset files in a `Dataset/data/` subdirectory within your project's root, or update the `DATA_DIR` path in the notebooks accordingly.

5.  **FastText Embeddings:**
    * The `deception-detection-group-88.ipynb` notebook includes a function to download and extract FastText embeddings (`crawl-300d-2M.vec.zip`) if they are not found locally.
    * The embeddings will be downloaded to the `OUTPUT_DIR` (defaults to `./output/` or `/kaggle/working/`). Ensure you have an internet connection when running this for the first time if the file isn't present.

6.  **Running the Notebooks:**
    * Launch Jupyter Notebook or JupyterLab:
        ```bash
        jupyter notebook
        # or
        jupyter lab
        ```
    * Open and run the cells in the desired notebook (e.g., `deception-detection-group-88.ipynb` for the main model).

## Training
The model was trained using:
* **Optimizer**: AdamW (Learning Rate=$8 \times 10^{-5}$, Weight Decay=$1 \times 10^{-4}$).
* **Loss**: Binary Cross-Entropy.
* **Metrics**: Accuracy, Recall (Lie=0), Precision (Lie=0).
* **Batch Size**: 32.
* **Epochs**: Maximum of 50.
* **Callbacks**: Early Stopping (monitoring `val_loss`, patience 10, restoring best weights), Model Checkpoint (saving best model based on `val_loss`).
* Training for the main model (`deception-detection-group-88.ipynb`) stopped after 16 epochs due to early stopping.

## Results

The primary results from the `deception-detection-group-88.ipynb` model on the held-out test set are:

| Metric             | Value   |
| ------------------ | ------- |
| Test Loss          | 0.4598  |
| **Test Accuracy** | **0.8581** |
| Precision (Lie=0)  | 0.2525  |
| Recall (Lie=0)     | 0.3167  |
| **F1-Score (Lie=0)**| **0.2810** |
| Precision (Truth=1)| 0.9328  |
| Recall (Truth=1)   | 0.9100  |
| F1-Score (Truth=1) | 0.9213  |
| **Macro F1-Score** | **0.6011** |
| Weighted F1-Score  | 0.8652  |


The model achieved high overall accuracy, mainly due to the majority 'Truth' class. The performance on the minority 'Lie' class (F1-score of 0.2810) showed improvement over baseline results.

**Classification Report (Test Set):**
          precision    recall  f1-score   support

Lie (0)     0.2525    0.3167    0.2810       240
Truth (1)     0.9328    0.9100    0.9213      2501

accuracy                         0.8581      2741
macro avg     0.5926    0.6134    0.6011      2741
weighted avg     0.8732    0.8581    0.8652      2741

A confusion matrix and training history plots (loss, accuracy, recall, precision) are generated by the `deception-detection-group-88.ipynb` notebook and saved in the `OUTPUT_DIR`. The confusion matrix visually confirms the difficulty in correctly identifying lies.

## Discussion & Conclusion

The hybrid model demonstrates improved performance over baselines, especially for the 'Lie' class (Lie F1-score of 0.2810), suggesting that combining Transformer-based text understanding with GCN-based contextual graph modeling captures relevant signals. However, the disparity between 'Truth' and 'Lie' class performance highlights the task's inherent difficulty. This is attributed to extreme class imbalance and the potential need for richer contextual features not captured by current scores and communication links.

The model achieved a Macro F1-score of 0.6011 and a Lie F1-score of 0.2810, representing tangible improvement. While challenges remain in robustly detecting the 'Lie' class, the results confirm the value of the hybrid architecture. Future work could focus on richer contextual features, advanced model components, fusion techniques, and alternative strategies for class imbalance. This work serves as a strong foundation in automated deception detection in strategic communication.

## Hyperparameters

Key hyperparameters for the main model (`deception-detection-group-88.ipynb`) are detailed below (and in Table 2 of the Report.pdf):

**Data/Preprocessing:**
* Max Sequence Length: 60
* Max Vocab Size: 10,000
* Embedding Dimension: 300
* Delta Feature Dimension: 2
* Max Players (Graph): 7
* Node Feature Dimension: 1

**Text Encoder:**
* Embedding Trainable: True
* Attention Heads (Text): 4
* Dropout Rate (Encoder): 0.15

**Graph Encoder:**
* GCN Units (L1, L2, L3): 64, 64, 32
* GCN Activation: ReLU
* Dropout Rate (Encoder): 0.15
* Final Graph Emb Dim: 150

**Fusion/Classification Head:**
* Attention Heads (Reason): 4
* Dropout Rate (Fusion): 0.4

**Training:**
* Optimizer: AdamW
* Learning Rate: <span class="math-inline">8 \\times 10^\{\-5\}</span>
* Weight Decay: <span class="math-inline">1 \\times 10^\{\-4\}</span>
* Batch Size: 32
* Max Epochs: 50
* Early Stopping Patience: 10
* Early Stopping Monitor: `val_loss`
