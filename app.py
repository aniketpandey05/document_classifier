import streamlit as st
import pickle
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


import os

from src.utils import preprocess_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

def load_nltk_data():
    download_nltk_resources()
    return stopwords.words('english'), WordNetLemmatizer()


@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, 'final_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'deployment_info.pkl'), 'rb') as f:
        info = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'active_learning_results.pkl'), 'rb') as f:
        results = pickle.load(f)
    return model, vectorizer, info, results


@st.cache_resource
def load_al_benchmark():
    """Multi-seed AL benchmark produced by `python -m src.al_benchmark` (None if not generated yet)."""
    path = os.path.join(BASE_DIR, 'al_benchmark_results.pkl')
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_train_test_data_matrices():
    with open(os.path.join(BASE_DIR, 'train_test_data.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']

@st.cache_resource
def load_sentence_embeddings():
    import numpy as np
    from src.utils import get_embeddings_model, load_raw_train_test_data
    
    train_emb_path = os.path.join(BASE_DIR, "X_train_emb.npy")
    test_emb_path = os.path.join(BASE_DIR, "X_test_emb.npy")
    
    if os.path.exists(train_emb_path) and os.path.exists(test_emb_path):
        X_train_emb = np.load(train_emb_path)
        X_test_emb = np.load(test_emb_path)
        return X_train_emb, X_test_emb
        
    encoder = get_embeddings_model()
    df_train, df_test = load_raw_train_test_data()
    
    X_train_emb = encoder.encode(df_train['text'].tolist(), show_progress_bar=True)
    X_test_emb = encoder.encode(df_test['text'].tolist(), show_progress_bar=True)
    
    np.save(train_emb_path, X_train_emb)
    np.save(test_emb_path, X_test_emb)
    
    return X_train_emb, X_test_emb

def recreate_model_instance(model_name, hyperparams):
    if model_name == "Logistic Regression":
        C_val = hyperparams['C']
        return LogisticRegression(C=C_val, max_iter=1000, solver='lbfgs', random_state=42)
    elif model_name == "Multinomial Naive Bayes":
        alpha_val = hyperparams['alpha']
        return MultinomialNB(alpha=alpha_val)
    elif model_name == "Linear SVM (SGD)":
        alpha_val = hyperparams['alpha']
        return SGDClassifier(loss='modified_huber', penalty='l2', alpha=alpha_val, random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

@st.cache_resource
def get_trained_model(feature_set, model_name, hyperparams, _X_train, _y_train):
    # _X_train/_y_train are not hashed (too large), so feature_set must be part
    # of the cache key or a TF-IDF model would be reused for embeddings.
    clf = recreate_model_instance(model_name, hyperparams)
    clf.fit(_X_train, _y_train)
    return clf


def main():
    st.set_page_config(
        page_title="Document Classifier with Active Learning",
        page_icon="📄",
        layout="wide"
    )

    # Inject Custom CSS for modern design aesthetics
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
            
            /* Apply global Outfit font */
            html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
                font-family: 'Outfit', sans-serif !important;
            }
            
            /* Custom styled metric card wrappers */
            div[data-testid="stMetric"] {
                background-color: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-radius: 12px !important;
                padding: 15px 20px !important;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
                transition: transform 0.2s, box-shadow 0.2s !important;
            }
            
            div[data-testid="stMetric"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05) !important;
                border-color: #3b82f6 !important;
            }

            /* Explicitly style metric labels and values to guarantee readability */
            div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
                color: #0f172a !important;
            }
            div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
                color: #475569 !important;
            }
            div[data-testid="stMetric"] label {
                color: #475569 !important;
            }

            /* Custom styled main header */
            .main-title {
                font-size: 2.5rem !important;
                font-weight: 700 !important;
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1.5rem;
                display: inline-block;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Feature Set Selection
    st.sidebar.header("🛠️ Feature Configuration")
    feature_set = st.sidebar.selectbox(
        "Feature Representation",
        ["TF-IDF Baseline", "Sentence Embeddings (MiniLM)"],
        help="Select the vector representation method to convert text to numbers."
    )

    # Load dataset matrices or dense embeddings
    X_train_full_mat, X_test_mat, y_train_full, y_test = load_train_test_data_matrices()
    
    if feature_set == "TF-IDF Baseline":
        X_train_full = X_train_full_mat
        X_test = X_test_mat
    else:
        X_train_full, X_test = load_sentence_embeddings()

    stop_words, lemmatizer = load_nltk_data()
    model_orig, vectorizer, info, results = load_models()
    categories = info['categories']
    benchmark = load_al_benchmark()
    if benchmark is not None:
        improvement = benchmark['best_gain_pp']
        improvement_help = (
            f"Best strategy ({benchmark['best_strategy']}) vs Random: mean final-accuracy "
            f"difference over {len(benchmark['config']['seeds'])} seeds."
        )
    else:
        improvement = results['improvement']
        improvement_help = "Uncertainty sampling vs Random, single run."

    # Sidebar Model Selection (filtered by feature compatibility)
    st.sidebar.header("🛠️ Classifier Configuration")
    if feature_set == "TF-IDF Baseline":
        model_options = ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM (SGD)"]
    else:
        model_options = ["Logistic Regression", "Linear SVM (SGD)"]
        st.sidebar.info("💡 Naive Bayes is disabled for dense embeddings (requires non-negative TF-IDF).")

    model_name = st.sidebar.selectbox(
        "Classifier Model",
        model_options,
        help="Select the machine learning algorithm to train on the dataset."
    )
    
    hyperparams = {}
    if model_name == "Logistic Regression":
        C_val = st.sidebar.slider(
            "Regularization Strength (C)", 
            min_value=0.01, 
            max_value=10.0, 
            value=1.0, 
            step=0.05, 
            help="Smaller values specify stronger regularization."
        )
        hyperparams['C'] = C_val
    elif model_name == "Multinomial Naive Bayes":
        alpha_val = st.sidebar.slider(
            "Smoothing Parameter (Alpha)", 
            min_value=0.01, 
            max_value=5.0, 
            value=1.0, 
            step=0.05, 
            help="Laplace smoothing parameter."
        )
        hyperparams['alpha'] = alpha_val
    elif model_name == "Linear SVM (SGD)":
        alpha_val = st.sidebar.select_slider(
            "Regularization Parameter (Alpha)", 
            options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 
            value=1e-4, 
            help="Multiplier for the regularization penalty."
        )
        hyperparams['alpha'] = alpha_val

    # Dynamically train model based on sliders
    model = get_trained_model(feature_set, model_name, hyperparams, X_train_full, y_train_full)
    
    # Evaluate custom model accuracy on test set
    y_pred_custom = model.predict(X_test)
    custom_accuracy = accuracy_score(y_test, y_pred_custom)

    st.markdown('<h1 class="main-title">📄 Document Classification with Active Learning</h1>', unsafe_allow_html=True)
    st.markdown("---")

    st.sidebar.header("📊 Current Model Stats")
    st.sidebar.metric("Test Accuracy", f"{custom_accuracy*100:.2f}%")
    st.sidebar.metric("Categories", len(categories))
    st.sidebar.metric("Training Samples", f"{X_train_full.shape[0]:,}")
    if feature_set == "TF-IDF Baseline":
        st.sidebar.metric("Vocabulary Size", f"{X_train_full.shape[1]:,}")
    else:
        st.sidebar.metric("Embedding Dimensions", f"{X_train_full.shape[1]:,}")

    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Active Learning Advantage")
    st.sidebar.metric(
        "Accuracy Gain vs Random",
        f"{improvement:+.2f}pp",
        help=improvement_help
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Classify Text", "⚙️ Labeling Studio", "🔍 Feature Explorer", "📈 Model Performance", "ℹ️ About"])

    with tab1:
        st.header("Classify a Document")

        user_input = st.text_area(
            "Enter text to classify:",
            height=200,
            placeholder="Paste your document here..."
        )

        if st.button("🚀 Classify", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                if feature_set == "TF-IDF Baseline":
                    cleaned = preprocess_text(user_input, stop_words, lemmatizer)
                    X = vectorizer.transform([cleaned])
                else:
                    from src.utils import get_embeddings_model
                    encoder = get_embeddings_model()
                    X = encoder.encode([user_input])
                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]

                st.success("Classification Complete")

                col1, col2 = st.columns([1, 2])

                with col1:
                    confidence = probabilities[prediction] * 100
                    st.markdown(f"### {categories[prediction]}")
                    st.metric("Confidence", f"{confidence:.1f}%")

                with col2:
                    prob_df = pd.DataFrame({
                        'Category': categories,
                        'Probability': probabilities * 100
                    }).sort_values('Probability', ascending=False)

                    top_prob_df = prob_df.head(10).copy()
                    chart = alt.Chart(top_prob_df).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('Probability:Q', title='Confidence (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('Category:N', sort='-x', title='Category'),
                        color=alt.Color('Probability:Q', scale=alt.Scale(scheme='blues', reverse=True), legend=None),
                        tooltip=[
                            alt.Tooltip('Category:N', title='Category'),
                            alt.Tooltip('Probability:Q', format='.2f', title='Probability (%)')
                        ]
                    ).properties(
                        title="Top 10 Category Probabilities",
                        height=300
                    )
                    st.altair_chart(chart, width='stretch')

                # Explain prediction - Word Highlights
                from src.utils import explain_prediction, explain_embeddings_prediction_loo, get_highlighted_html
                with st.spinner("Analyzing feature contributions..."):
                    if feature_set == "TF-IDF Baseline":
                        word_contributions = explain_prediction(
                            user_input, vectorizer, model, prediction, stop_words, lemmatizer
                        )
                    else:
                        from src.utils import get_embeddings_model
                        encoder = get_embeddings_model()
                        word_contributions = explain_embeddings_prediction_loo(
                            user_input, encoder, model, prediction, stop_words, lemmatizer
                        )
                    highlighted_html = get_highlighted_html(
                        user_input, word_contributions, lemmatizer
                    )

                st.markdown("---")
                st.markdown("### 🔍 Model Explanation (Word Contributions)")
                st.markdown(
                    "The highlighted words below drove the model's classification decision. "
                    "Hover over any highlighted word to inspect its numeric contribution score. "
                    "**Green** words increased confidence in this category; **red** words decreased it."
                )
                
                # HTML Container
                st.markdown(
                    f'<div style="background-color: #f8f9fa; color: #0f172a; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; font-family: monospace; line-height: 1.8; white-space: pre-wrap; font-size: 15px;">{highlighted_html}</div>',
                    unsafe_allow_html=True
                )

        # Batch classification section
        st.markdown("---")
        st.subheader("📂 Batch Classification (CSV Upload)")
        st.markdown(
            "Upload a CSV file containing multiple documents. Select the column containing "
            "the text, and download the prediction results as a new CSV file."
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            columns = df_upload.columns.tolist()
            text_col = st.selectbox("Select the column containing document text:", columns)

            if st.button("🚀 Classify Batch", type="primary"):
                if df_upload[text_col].isnull().any():
                    st.warning("Warning: Selected column contains empty/null rows. These will be classified as empty.")

                total_rows = len(df_upload)
                preds = [None] * total_rows
                confidences = [None] * total_rows

                # Pre-identify empty/nan rows
                valid_indices = []
                valid_texts = []
                for idx, row in df_upload.iterrows():
                    text_val = str(row[text_col])
                    if not text_val.strip() or text_val == "nan":
                        preds[idx] = "N/A"
                        confidences[idx] = 0.0
                    else:
                        valid_indices.append(idx)
                        valid_texts.append(text_val)

                if valid_texts:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    if feature_set == "TF-IDF Baseline":
                        cleaned_texts = []
                        for i, text_val in enumerate(valid_texts):
                            status_text.text(f"Preprocessing row {i + 1}/{len(valid_texts)}...")
                            cleaned_texts.append(preprocess_text(text_val, stop_words, lemmatizer))
                            progress_bar.progress((i + 1) / len(valid_texts))

                        status_text.text("Classifying documents...")
                        X_vals = vectorizer.transform(cleaned_texts)
                        pred_vals = model.predict(X_vals)
                        prob_vals = model.predict_proba(X_vals)

                        for i, idx in enumerate(valid_indices):
                            pred_idx = pred_vals[i]
                            preds[idx] = categories[pred_idx]
                            confidences[idx] = prob_vals[i][pred_idx] * 100
                    else:
                        status_text.text("Generating embeddings for batch...")
                        from src.utils import get_embeddings_model
                        encoder = get_embeddings_model()
                        X_vals = encoder.encode(valid_texts, show_progress_bar=False)

                        status_text.text("Classifying documents...")
                        pred_vals = model.predict(X_vals)
                        prob_vals = model.predict_proba(X_vals)

                        for i, idx in enumerate(valid_indices):
                            pred_idx = pred_vals[i]
                            prob_val = prob_vals[i][pred_idx]
                            preds[idx] = categories[pred_idx]
                            confidences[idx] = prob_val * 100
                            progress_bar.progress((idx + 1) / total_rows)

                    progress_bar.empty()
                    status_text.success("Batch classification complete!")

                df_results = df_upload.copy()
                df_results['Predicted Category'] = preds
                df_results['Confidence (%)'] = confidences

                col_stats1, col_stats2 = st.columns([1, 1])
                with col_stats1:
                    st.markdown("#### Preview Predictions")
                    st.dataframe(df_results.head(10))

                with col_stats2:
                    st.markdown("#### Category Distribution")
                    dist_df = df_results['Predicted Category'].value_counts().reset_index()
                    dist_df.columns = ['Category', 'Count']

                    dist_chart = alt.Chart(dist_df).mark_bar(cornerRadiusEnd=4).encode(
                        x=alt.X('Count:Q', title='Document Count'),
                        y=alt.Y('Category:N', sort='-x', title='Category'),
                        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues', reverse=True), legend=None),
                        tooltip=['Category', 'Count']
                    ).properties(height=250)
                    st.altair_chart(dist_chart, width='stretch')

                csv_data = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="classified_documents.csv",
                    mime="text/csv"
                )

    with tab2:
        st.header("⚙️ Active Learning Labeling Studio")
        st.markdown(
            "Act as the **Human Oracle**! Label the documents that the model is **most uncertain** about "
            "based on your selected sampling strategy. The model will retrain instantly with your feedback."
        )

        # Check if the feature set, model selection, or hyperparameters changed, and update/reinitialize
        if 'al_initialized' in st.session_state:
            if st.session_state.get('al_feature_set') != feature_set:
                for key in list(st.session_state.keys()):
                    if key.startswith('al_'):
                        del st.session_state[key]
            elif (st.session_state.get('al_model_name') != model_name or 
                  st.session_state.get('al_hyperparams') != hyperparams):
                
                with st.spinner("Updating Active Learning model configuration..."):
                    model_al = recreate_model_instance(model_name, hyperparams)
                    model_al.fit(st.session_state['al_X_train'], st.session_state['al_y_train'])
                    st.session_state['al_model'] = model_al
                    
                    st.session_state['al_model_name'] = model_name
                    st.session_state['al_hyperparams'] = hyperparams
                    
                    # Update initial history evaluation using selected model
                    y_pred_init = recreate_model_instance(model_name, hyperparams)
                    X_init = st.session_state['al_X_train'][:500]
                    y_init = st.session_state['al_y_train'][:500]
                    y_pred_init.fit(X_init, y_init)
                    init_acc = accuracy_score(st.session_state['y_test'], y_pred_init.predict(st.session_state['X_test']))
                    init_f1 = f1_score(st.session_state['y_test'], y_pred_init.predict(st.session_state['X_test']), average='weighted')
                    st.session_state['al_history'][0] = {'n_labeled': 500, 'accuracy': init_acc, 'f1': init_f1}
                    
                    # Update current evaluation
                    y_pred_cur = model_al.predict(st.session_state['X_test'])
                    cur_acc = accuracy_score(st.session_state['y_test'], y_pred_cur)
                    cur_f1 = f1_score(st.session_state['y_test'], y_pred_cur, average='weighted')
                    st.session_state['al_history'][-1] = {
                        'n_labeled': len(st.session_state['al_y_train']),
                        'accuracy': cur_acc,
                        'f1': cur_f1
                    }

        # Initialize AL session state if not already done
        if 'al_initialized' not in st.session_state:
            with st.spinner("Initializing Active Learning Studio (loading datasets)..."):
                from src.utils import load_raw_train_test_data
                
                # Use current representation matrices
                st.session_state['X_train_full'] = X_train_full
                st.session_state['y_train_full'] = y_train_full
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                
                # Load raw text data
                df_train, df_test = load_raw_train_test_data()
                st.session_state['df_train'] = df_train
                
                # Pick 500 random samples as the starting labeled set
                np.random.seed(42)
                initial_indices = np.random.choice(len(df_train), size=500, replace=False)
                pool_indices = [i for i in range(len(df_train)) if i not in initial_indices]
                
                # Slice matrices
                X_labeled = st.session_state['X_train_full'][initial_indices]
                y_labeled = st.session_state['y_train_full'][initial_indices]
                
                # Train baseline model for active learning
                model_al = recreate_model_instance(model_name, hyperparams)
                model_al.fit(X_labeled, y_labeled)
                
                # Evaluate
                y_pred = model_al.predict(st.session_state['X_test'])
                acc = accuracy_score(st.session_state['y_test'], y_pred)
                f1 = f1_score(st.session_state['y_test'], y_pred, average='weighted')
                
                st.session_state['al_X_train'] = X_labeled
                st.session_state['al_y_train'] = y_labeled
                st.session_state['al_pool_indices'] = pool_indices
                st.session_state['al_model'] = model_al
                st.session_state['al_model_name'] = model_name
                st.session_state['al_hyperparams'] = hyperparams
                st.session_state['al_feature_set'] = feature_set
                st.session_state['al_history'] = [{'n_labeled': 500, 'accuracy': acc, 'f1': f1}]
                st.session_state['al_current_sample_idx'] = None
                st.session_state['al_labeled_count'] = 0
                st.session_state['al_initialized'] = True

        # Controls
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 3])
        with col_ctrl1:
            strategy = st.selectbox(
                "Sampling Strategy",
                ["Entropy", "Margin", "Random"],
                help="Entropy: Selects sample with highest prediction uncertainty across all classes.\n"
                     "Margin: Selects sample with smallest confidence margin between top 2 classes.\n"
                     "Random: Selects a random sample from the pool."
            )
        with col_ctrl2:
            st.metric("Samples Labeled in Session", st.session_state['al_labeled_count'])
        with col_ctrl3:
            current_acc = st.session_state['al_history'][-1]['accuracy'] * 100
            initial_acc = st.session_state['al_history'][0]['accuracy'] * 100
            gain = current_acc - initial_acc
            st.metric(
                "Live Model Accuracy", 
                f"{current_acc:.2f}%", 
                delta=f"+{gain:.2f}pp" if gain >= 0 else f"{gain:.2f}pp"
            )

        # Select sample
        if st.session_state['al_current_sample_idx'] is None:
            if len(st.session_state['al_pool_indices']) > 0:
                X_train_full = st.session_state['X_train_full']
                pool_indices = st.session_state['al_pool_indices']
                model_al = st.session_state['al_model']
                
                if strategy == "Random":
                    next_idx = int(np.random.choice(pool_indices))
                else:
                    X_pool = X_train_full[pool_indices]
                    probas = model_al.predict_proba(X_pool)
                    
                    if strategy == "Entropy":
                        entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
                        relative_idx = np.argmax(entropy)
                    elif strategy == "Margin":
                        sorted_probas = np.sort(probas, axis=1)
                        margin = sorted_probas[:, -1] - sorted_probas[:, -2]
                        relative_idx = np.argmin(margin)
                    
                    next_idx = pool_indices[relative_idx]
                
                st.session_state['al_current_sample_idx'] = next_idx
            else:
                st.info("Congratulations! All pool samples have been labeled.")
                st.session_state['al_current_sample_idx'] = None

        # Display active sample
        current_idx = st.session_state['al_current_sample_idx']
        if current_idx is not None:
            df_train = st.session_state['df_train']
            raw_text = df_train.iloc[current_idx]['text']
            true_cat = df_train.iloc[current_idx]['category']
            
            X_sample = st.session_state['X_train_full'][current_idx]
            if feature_set != "TF-IDF Baseline":
                X_sample = X_sample.reshape(1, -1)
            model_al = st.session_state['al_model']
            pred_idx = model_al.predict(X_sample)[0]
            pred_prob = model_al.predict_proba(X_sample)[0][pred_idx] * 100
            pred_cat = categories[pred_idx]

            st.markdown("### 📄 Document to Label")
            
            st.text_area(
                "Document Text (Unlabeled)",
                value=raw_text,
                height=250,
                disabled=True
            )

            st.markdown(f"🤖 **Model's Current Prediction:** `{pred_cat}` (Confidence: {pred_prob:.1f}%)")

            try:
                default_sel_idx = categories.index(pred_cat)
            except ValueError:
                default_sel_idx = 0

            with st.form("labeling_form", clear_on_submit=True):
                selected_label_str = st.selectbox(
                    "Assign the correct category:",
                    categories,
                    index=default_sel_idx
                )
                
                submitted = st.form_submit_button("🚀 Submit Label & Retrain Model", type="primary")
                
                if submitted:
                    selected_label_idx = categories.index(selected_label_str)
                    
                    if feature_set == "TF-IDF Baseline":
                        st.session_state['al_X_train'] = vstack([st.session_state['al_X_train'], X_sample])
                    else:
                        st.session_state['al_X_train'] = np.vstack([st.session_state['al_X_train'], X_sample])
                    st.session_state['al_y_train'] = np.append(st.session_state['al_y_train'], selected_label_idx)
                    st.session_state['al_pool_indices'].remove(current_idx)
                    
                    st.session_state['al_model'].fit(st.session_state['al_X_train'], st.session_state['al_y_train'])
                    
                    y_pred = st.session_state['al_model'].predict(st.session_state['X_test'])
                    new_acc = accuracy_score(st.session_state['y_test'], y_pred)
                    new_f1 = f1_score(st.session_state['y_test'], y_pred, average='weighted')
                    
                    st.session_state['al_history'].append({
                        'n_labeled': len(st.session_state['al_y_train']),
                        'accuracy': new_acc,
                        'f1': new_f1
                    })
                    
                    st.session_state['al_labeled_count'] += 1
                    st.session_state['al_current_sample_idx'] = None
                    
                    st.success(f"Feedback recorded! Retrained model. New Accuracy: {new_acc*100:.2f}%")
                    st.rerun()

    with tab3:
        st.header("🔍 Category Feature Explorer")
        if feature_set != "TF-IDF Baseline":
            st.info(
                "💡 **Feature Explorer Limitation for Dense Embeddings**\n\n"
                "For dense sentence embeddings (MiniLM), model coefficients correspond to dimensions in the "
                "384-dimensional dense semantic vector space rather than individual words in a vocabulary.\n\n"
                "Because these dimensions are combinations of latent semantic concepts, they cannot be directly mapped "
                "back to raw keywords.\n\n"
                "To explore feature contributions for embeddings, use the **🔮 Classify Text** tab, which uses a custom "
                "**Leave-One-Out (LOO)** perturbation-based approach to highlight word importance on-the-fly!"
            )
        else:
            st.markdown(
                "Inspect the top **15 words** that are most strongly associated with each class based on the model's coefficients. "
                "This highlights what features (keywords) the classifier has learned to value most for making its decisions."
            )

            sel_cat_str = st.selectbox(
                "Select Category to Explore:",
                categories,
                index=0,
                key="feature_explorer_category_select"
            )
            sel_cat_idx = categories.index(sel_cat_str)

            from src.utils import get_top_features_for_category
            top_feat_df = get_top_features_for_category(model, vectorizer, sel_cat_idx, top_n=15)

            if not top_feat_df.empty:
                top_feat_df = top_feat_df.sort_values(by="Weight", ascending=True)
                weight_title = "Log-Odds vs Other Classes" if model_name == "Multinomial Naive Bayes" else "Coefficient (Weight)"
                
                chart_feat = alt.Chart(top_feat_df).mark_bar(cornerRadiusEnd=4).encode(
                    x=alt.X('Weight:Q', title=weight_title),
                    y=alt.Y('Word:N', sort='-x', title='Word'),
                    color=alt.Color('Weight:Q', scale=alt.Scale(scheme='blues', reverse=True), legend=None),
                    tooltip=['Word', alt.Tooltip('Weight:Q', format='.4f')]
                ).properties(
                    title=f"Top Influential Words for '{sel_cat_str}'",
                    height=450
                )
                st.altair_chart(chart_feat, width='stretch')
            else:
                st.warning("Model does not support coefficient extraction.")

    with tab4:
        st.header("📈 Model Performance Analysis")
        st.markdown(
            "Compare the learning efficiency of **Active Learning** vs. **Random Sampling**. "
            "If you have labeled samples in this session, your **live session** performance will also be shown!"
        )

        al_df = results['active_learning'].copy()
        random_df = results['random_sampling'].copy()

        al_df['Strategy'] = 'Active Learning (Pre-calculated)'
        al_df['Accuracy (%)'] = al_df['accuracy'] * 100
        al_df['F1-Score (%)'] = al_df['f1'] * 100

        random_df['Strategy'] = 'Random Sampling (Pre-calculated)'
        random_df['Accuracy (%)'] = random_df['accuracy'] * 100
        random_df['F1-Score (%)'] = random_df['f1'] * 100

        combined_df = pd.concat([al_df, random_df], ignore_index=True)

        if 'al_history' in st.session_state and len(st.session_state['al_history']) > 1:
            session_df = pd.DataFrame(st.session_state['al_history'])
            session_df['Strategy'] = 'Interactive Session (Live)'
            session_df['Accuracy (%)'] = session_df['accuracy'] * 100
            session_df['F1-Score (%)'] = session_df['f1'] * 100
            combined_df = pd.concat([combined_df, session_df], ignore_index=True)

        col1, col2 = st.columns(2)

        with col1:
            acc_chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X('n_labeled:Q', title='Number of Labeled Samples'),
                y=alt.Y('Accuracy (%):Q', title='Accuracy (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Strategy:N', scale=alt.Scale(
                    domain=['Active Learning (Pre-calculated)', 'Random Sampling (Pre-calculated)', 'Interactive Session (Live)'],
                    range=['#3b82f6', '#9ca3af', '#10b981']
                )),
                tooltip=['Strategy', 'n_labeled', alt.Tooltip('Accuracy (%):Q', format='.2f')]
            ).properties(
                title="Model Accuracy Comparison",
                height=350
            ).interactive()
            st.altair_chart(acc_chart, width='stretch')

        with col2:
            f1_chart = alt.Chart(combined_df).mark_line(point=True).encode(
                x=alt.X('n_labeled:Q', title='Number of Labeled Samples'),
                y=alt.Y('F1-Score (%):Q', title='F1-Score (%)', scale=alt.Scale(zero=False)),
                color=alt.Color('Strategy:N', scale=alt.Scale(
                    domain=['Active Learning (Pre-calculated)', 'Random Sampling (Pre-calculated)', 'Interactive Session (Live)'],
                    range=['#3b82f6', '#9ca3af', '#10b981']
                )),
                tooltip=['Strategy', 'n_labeled', alt.Tooltip('F1-Score (%):Q', format='.2f')]
            ).properties(
                title="Model F1-Score Comparison",
                height=350
            ).interactive()
            st.altair_chart(f1_chart, width='stretch')

        st.markdown("---")
        st.subheader("🧪 Multi-Seed Strategy Benchmark")
        if benchmark is None:
            st.info("Run `python -m src.al_benchmark` to generate the multi-seed benchmark.")
        else:
            cfg = benchmark['config']
            st.markdown(
                f"Each strategy was run with **{len(cfg['seeds'])} different seeds** "
                f"(different starting labeled sets), {cfg['n_rounds']} rounds × {cfg['batch_size']} labels, "
                f"using {cfg['model']} on {cfg['features']}. Lines show the **mean** accuracy and the shaded "
                "band shows **± 1 standard deviation** across seeds. If two bands overlap heavily, the "
                "difference between those strategies is within run-to-run noise."
            )

            band_df = benchmark['curves'].groupby(['strategy', 'n_labeled'])['accuracy'].agg(['mean', 'std']).reset_index()
            band_df['Mean (%)'] = band_df['mean'] * 100
            band_df['Lower'] = (band_df['mean'] - band_df['std']) * 100
            band_df['Upper'] = (band_df['mean'] + band_df['std']) * 100
            band_df = band_df.rename(columns={'strategy': 'Strategy'})

            strategy_color = alt.Color('Strategy:N', sort=cfg['strategies'])
            band = alt.Chart(band_df).mark_area(opacity=0.15).encode(
                x=alt.X('n_labeled:Q', title='Number of Labeled Samples'),
                y=alt.Y('Lower:Q', title='Accuracy (%)', scale=alt.Scale(zero=False)),
                y2='Upper:Q',
                color=strategy_color
            )
            line = alt.Chart(band_df).mark_line(point=True).encode(
                x='n_labeled:Q',
                y='Mean (%):Q',
                color=strategy_color,
                tooltip=['Strategy', 'n_labeled',
                         alt.Tooltip('Mean (%):Q', format='.2f'),
                         alt.Tooltip('std:Q', format='.4f', title='Std (fraction)')]
            )
            st.altair_chart(
                (band + line).properties(title="Accuracy vs Labels (mean ± std over seeds)", height=400).interactive(),
                width='stretch'
            )

            summary = benchmark['summary']
            table = pd.DataFrame({
                'Final Accuracy (%)': [f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(summary['final_acc_mean'], summary['final_acc_std'])],
                'Final Macro-F1 (%)': (summary['final_macro_f1_mean'] * 100).round(2),
                'AULC (%)': [f"{m*100:.2f} ± {s*100:.2f}" for m, s in zip(summary['aulc_mean'], summary['aulc_std'])],
                'Gain vs Random (pp)': [
                    "—" if pd.isna(m) else f"{m:+.2f} ± {s:.2f}"
                    for m, s in zip(summary['mean_gain_pp'], summary['std_gain_pp'])
                ],
                'Seeds Beating Random': [
                    "—" if pd.isna(v) else f"{int(v)}/{len(cfg['seeds'])}" for v in summary['seeds_beating_random']
                ],
            }, index=summary.index)
            st.dataframe(table, width='stretch')
            st.caption(
                "AULC = area under the learning curve (mean accuracy over all rounds); it rewards strategies "
                "that learn fast early, not just at the end. Gain vs Random is paired by seed, so both runs "
                "start from the same labeled set."
            )

        st.markdown("---")
        st.subheader("🎯 Confusion Matrix Heatmap")
        st.markdown("Hover over the squares to inspect how categories are confused by the baseline model on the test set.")

        with st.spinner("Computing Confusion Matrix..."):
            from sklearn.metrics import confusion_matrix
            y_pred_base = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_base)

            cm_data = []
            for i in range(len(categories)):
                for j in range(len(categories)):
                    cm_data.append({
                        'Actual': categories[i],
                        'Predicted': categories[j],
                        'Count': int(cm[i, j])
                    })
            cm_df = pd.DataFrame(cm_data)

            cm_chart = alt.Chart(cm_df).mark_rect().encode(
                x=alt.X('Predicted:N', title='Predicted Category', sort=categories, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Actual:N', title='Actual Category', sort=categories),
                color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Number of Docs'),
                tooltip=['Actual', 'Predicted', 'Count']
            ).properties(
                title="Confusion Matrix Heatmap (Baseline Model)",
                height=500
            )
            st.altair_chart(cm_chart, width='stretch')

    with tab5:
        st.header("About")
        st.write(
            "This project demonstrates Active Learning for text classification "
            "using the 20 Newsgroups dataset. It supports both classical sparse TF-IDF features "
            "and dense semantic embeddings generated via Sentence Transformers (all-MiniLM-L6-v2)."
        )
        st.write(
            "For model interpretability, it features a TF-IDF coefficient mapper as well as a "
            "Leave-One-Out (LOO) perturbation explainer to identify word-level feature contributions "
            "for dense vector representations on-the-fly."
        )



if __name__ == "__main__":
    main()
