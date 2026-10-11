import re
import nltk
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from nltk.tokenize import word_tokenize

def preprocess_text(text, stop_words, lemmatizer):
    """
    Cleans and preprocesses the input text.
    Removes URLs, emails, numbers, punctuation, tokenizes, removes stopwords, and lemmatizes.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_raw_train_test_data():
    """
    Loads preprocessed_data.pkl and splits it in the exact same way as feature_engineering.ipynb.
    This guarantees 1-to-1 correspondence with the row indices in X_train_tfidf.
    """
    df = pd.read_pickle(os.path.join(BASE_DIR, 'preprocessed_data.pkl'))
    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

def explain_prediction(text, vectorizer, model, pred_class_idx, stop_words, lemmatizer):
    """
    Calculates the contribution of each word in the input text towards the predicted class.
    Contribution = word_tfidf_value * class_coefficient/log-odds.
    """
    cleaned = preprocess_text(text, stop_words, lemmatizer)
    words = cleaned.split()
    if not words:
        return {}
    
    # Vectorize input text
    X = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    
    # Determine the coefficients dynamically based on model attributes
    if hasattr(model, 'feature_log_prob_'):
        # MultinomialNB: Calculate log-odds of the predicted class vs the average of other classes
        log_probs = model.feature_log_prob_
        target_log_prob = log_probs[pred_class_idx]
        n_classes = log_probs.shape[0]
        other_log_prob_sum = np.sum(log_probs, axis=0) - target_log_prob
        mean_other_log_prob = other_log_prob_sum / (n_classes - 1)
        class_coefs = target_log_prob - mean_other_log_prob
    elif hasattr(model, 'coef_'):
        # Linear models (LogisticRegression, SGDClassifier)
        coefs = model.coef_
        if len(coefs.shape) > 1 and coefs.shape[0] > 1:
            class_coefs = coefs[pred_class_idx]
        else:
            class_coefs = coefs[0] if pred_class_idx == 1 else -coefs[0]
    else:
        return {}
        
    word_contributions = {}
    
    # Extract non-zero TF-IDF features from the sparse row
    coo = X.tocoo()
    for col, val in zip(coo.col, coo.data):
        word = feature_names[col]
        coef_val = class_coefs[col]
        contribution = coef_val * val
        word_contributions[word] = contribution
        
    return word_contributions

def get_highlighted_html(original_text, word_contributions, lemmatizer):
    """
    Generates HTML string with words highlighted based on their contributions.
    Positive contributions are highlighted in green, negative in red.
    The intensity corresponds to the relative contribution size.
    """
    if not word_contributions:
        return original_text
        
    # Split text into tokens including whitespace to preserve structure
    tokens = re.split(r'(\s+)', original_text)
    
    max_contrib = max([abs(v) for v in word_contributions.values()]) if word_contributions else 1.0
    if max_contrib == 0:
        max_contrib = 1.0
        
    highlighted_tokens = []
    
    for token in tokens:
        if not token.strip():
            highlighted_tokens.append(token)
            continue
            
        # Extract word character base for lookup
        word_clean = re.sub(r'[^a-zA-Z]', '', token).lower()
        if not word_clean:
            highlighted_tokens.append(token)
            continue
            
        word_lemmatized = lemmatizer.lemmatize(word_clean)
        contrib = word_contributions.get(word_lemmatized, 0)
        
        if contrib != 0:
            norm_contrib = contrib / max_contrib
            alpha = 0.15 + 0.7 * min(abs(norm_contrib), 1.0)
            
            if contrib > 0:
                color = f"rgba(46, 204, 113, {alpha:.2f})"
                text_color = "#145a32" if alpha > 0.4 else "#196f3d"
                title = f"Word: '{word_lemmatized}' | Contribution: +{contrib:.4f} (Positive)"
            else:
                color = f"rgba(231, 76, 60, {alpha:.2f})"
                text_color = "#641e16" if alpha > 0.4 else "#7b241c"
                title = f"Word: '{word_lemmatized}' | Contribution: {contrib:.4f} (Negative)"
                
            highlighted_tokens.append(
                f'<span style="background-color: {color}; color: {text_color}; padding: 2px 4px; border-radius: 4px; font-weight: 500; cursor: help;" title="{title}">{token}</span>'
            )
        else:
            highlighted_tokens.append(token)
            
    return "".join(highlighted_tokens)

def get_top_features_for_category(model, vectorizer, category_idx, top_n=15):
    """
    Retrieves the top N features (words) with the highest weights/probabilities 
    for a given category index and model.
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Check if the model is MultinomialNB
    if hasattr(model, 'feature_log_prob_'):
        # Rank by log-odds vs the other classes; raw log-probs would just surface
        # words that are common in every class (e.g. "would", "one").
        log_probs = model.feature_log_prob_
        n_classes = log_probs.shape[0]
        mean_other = (np.sum(log_probs, axis=0) - log_probs[category_idx]) / (n_classes - 1)
        coefs = log_probs[category_idx] - mean_other
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if len(coefs.shape) > 1 and coefs.shape[0] > 1:
            coefs = coefs[category_idx]
        else:
            coefs = coefs[0] if category_idx == 1 else -coefs[0]
    else:
        return pd.DataFrame(columns=['Word', 'Weight'])
        
    top_indices = np.argsort(coefs)[-top_n:][::-1]
    
    top_words = [feature_names[idx] for idx in top_indices]
    top_weights = [float(coefs[idx]) for idx in top_indices]
    
    return pd.DataFrame({
        'Word': top_words,
        'Weight': top_weights
    })

def get_embeddings_model():
    """
    Lazy-loads the lightweight SentenceTransformer model.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

def explain_embeddings_prediction_loo(text, encoder, model, pred_class_idx, stop_words, lemmatizer):
    """
    Calculates the contribution of each word in the input text towards the predicted class 
    using a Leave-One-Out (LOO) perturbation method for dense embeddings.
    """
    # Split text into tokens keeping whitespace structure
    tokens = re.split(r'(\s+)', text)
    
    # Extract unique words
    clean_words = []
    for token in tokens:
        if not token.strip():
            continue
        word_clean = re.sub(r'[^a-zA-Z]', '', token).lower()
        if word_clean:
            clean_words.append(word_clean)
            
    unique_words = list(set(clean_words))
    if not unique_words:
        return {}
        
    # Baseline probability on the full text
    base_vector = encoder.encode([text])
    base_prob = model.predict_proba(base_vector)[0][pred_class_idx]
    
    word_contributions = {}
    
    # Perturbation analysis: remove each word and compute the probability drop
    for word in unique_words:
        word_lem = lemmatizer.lemmatize(word)
        
        perturbed_tokens = []
        for token in tokens:
            word_clean = re.sub(r'[^a-zA-Z]', '', token).lower()
            if word_clean and lemmatizer.lemmatize(word_clean) == word_lem:
                # Omit occurrences of the word
                continue
            perturbed_tokens.append(token)
            
        perturbed_text = "".join(perturbed_tokens)
        
        if not perturbed_text.strip():
            perturbed_prob = 0.0
        else:
            perturbed_vector = encoder.encode([perturbed_text])
            perturbed_prob = model.predict_proba(perturbed_vector)[0][pred_class_idx]
            
        # Contribution is the drop in probability
        word_contributions[word_lem] = float(base_prob - perturbed_prob)
        
    return word_contributions
