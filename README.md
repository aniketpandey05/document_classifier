# Document Classifier with Active Learning & Local Explanations

This is a Streamlit web application that demonstrates **Active Learning** (Human-in-the-Loop) and **Explainable AI (XAI)** on the 20 Newsgroups text classification dataset. The project supports both classic sparse TF-IDF features and dense semantic sentence embeddings.

---

## 🌟 Key Features

### 1. Local Word Explanations
*   **TF-IDF Explanations:** Highlights word contributions using model coefficients (or log-odds ratios for Naive Bayes).
*   **Embedding Explanations:** Uses a custom **Leave-One-Out (LOO)** perturbation method. It sequentially removes each token from the text and measures the drop in prediction probability to calculate word importance.
*   **Visual Highlights:** Green words increase confidence in the predicted class, while red words decrease it. You can hover over any highlighted word to inspect its numeric contribution score.

### 2. Live Model Configuration
*   **Multiple Classifiers:** Swap between **Logistic Regression**, **Multinomial Naive Bayes**, and a **Linear SVM** (trained via SGD with modified Huber loss).
*   **Hyperparameter Tuning:** Adjust parameters (like regularization strength $C$ or Laplace smoothing $\alpha$) using sidebar sliders and watch the model retrain in milliseconds.
*   **Embedding Constraints:** The app automatically disables Multinomial Naive Bayes when using dense sentence embeddings because embeddings contain negative values.

### 3. Active Learning Labeling Studio
*   **Human-in-the-Loop:** Act as the oracle to label documents that the model is most uncertain about.
*   **Sampling Strategies:** Choose between **Entropy**, **Margin** uncertainty, or **Random** sampling.
*   **Live Retraining:** Once you submit a label, the model retrains instantly, updates the metrics, and plots your progress against pre-calculated benchmarks.
*   **Auto-Reset:** Switching feature sets automatically resets the labeling studio to avoid shape mismatch issues.

### 4. Category Feature Explorer
*   **Word Associations:** Select a class to see the top 15 words most strongly associated with it based on model coefficients (available when TF-IDF is active).
*   **Latent Space Explanation:** Shows an educational notice when embeddings are active, explaining why 384-dimensional dense semantic vectors cannot be mapped directly back to single words.

### 5. Batch Classification
*   **CSV Upload:** Process batches of documents by uploading a CSV.
*   **Optimized Inference:** Uses fast batch encoding for Sentence Transformers to maximize processing speed.
*   **Export:** Preview predictions, inspect the predicted category distribution, and download the results as a CSV.

### 6. Model Performance Dashboard
*   **Learning Curves:** Accuracy and F1 vs. number of labels for active learning, random sampling and your live labeling session.
*   **Multi-Seed Benchmark:** Mean ± std learning curves for five sampling strategies over five seeds, with a summary table.
*   **Confusion Matrix:** Interactive heatmap showing which categories the model confuses.

---

## 📊 Results

**Full training set (8,795 labeled docs, TF-IDF, default settings):**

| Model | Test Accuracy |
|-------|--------------:|
| Logistic Regression | 72.26% |
| Multinomial Naive Bayes | 70.67% |

**Active learning benchmark** (TF-IDF + Logistic Regression, 879 → 1,829 labels, 5 seeds, mean ± std):

| Strategy | Final Accuracy (%) | Gain vs Random (pp) | Seeds Beating Random |
|----------|---------------:|---------------:|:--------------------:|
| Random | 63.86 ± 0.82 | — | — |
| Least Confidence | 63.44 ± 0.73 | −0.42 | 1/5 |
| Margin | 63.38 ± 1.02 | −0.47 | 1/5 |
| Entropy | 62.83 ± 0.59 | −1.03 | 0/5 |
| **Diverse Margin** | **64.07 ± 0.43** | **+0.22** | **4/5** |

Pure uncertainty sampling did not beat random sampling on this dataset. Adding batch diversity (clustering the most uncertain documents and labeling one per cluster) was the only strategy to come out ahead, though the gain is within run-to-run variance.

---

## 📁 Project Structure

```
document_classifier/
├── app.py                      # Streamlit app (UI + orchestration)
├── src/
│   ├── utils.py                # Preprocessing, explanations, highlighting, embeddings
│   └── al_benchmark.py         # Multi-seed active learning benchmark
├── notebooks/
│   ├── explore_data.ipynb      # Class balance and sample documents
│   ├── text_preprocess.ipynb   # Text cleaning -> preprocessed_data.pkl
│   ├── feature_engineering.ipynb  # Train/test split + TF-IDF
│   ├── baseline_model.ipynb    # Logistic Regression on 10% of labels
│   ├── active_learning.ipynb   # Original single-run AL experiment
│   └── analysis.ipynb          # Plots, reports, exported model
├── *.pkl                       # Saved data, vectorizer, models and results
├── requirements.txt
└── run.bat                     # Windows launcher using the project venv
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rogueslasher/document_classifier.git
    cd document_classifier
    ```

2.  **Activate the Virtual Environment:**
    *   On Windows:
        ```powershell
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Web App:**
    ```bash
    streamlit run app.py
    ```

5.  **(Optional) Re-run the Multi-Seed Active Learning Benchmark:**
    ```bash
    python -m src.al_benchmark
    ```
    This compares Random, Least Confidence, Margin, Entropy and Diverse Margin sampling over 5 seeds and writes `al_benchmark_results.pkl`, which the **Model Performance** tab displays.

---

## 🔬 Technical Details

*   **Dataset:** 20 Newsgroups corpus with headers, footers and quotes removed (10,994 documents after cleaning, stratified 80/20 train/test split).
*   **Feature Representations:**
    *   *TF-IDF Baseline:* Sparse vectors (5,000 max features, unigrams, min document frequency = 5).
    *   *Sentence Embeddings:* Dense vectors from `all-MiniLM-L6-v2` (cached locally to `X_train_emb.npy` and `X_test_emb.npy` after the first run to allow instant subsequent loading).
*   **Models:** Logistic Regression (L-BFGS), SGDClassifier (linear SVM), and Multinomial Naive Bayes.
