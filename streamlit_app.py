import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
from pathlib import Path
from joblib import dump, load
from wordcloud import WordCloud

# Download only the NLTK data we still need
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Page Configuration
st.set_page_config(
    page_title="Amazon Musical Instrument Review Sentiment Analysis",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reduce spacing between sections
st.markdown("""
<style>
    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 0.4rem;
    }

    h1, h2, h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.35rem !important;
    }

    div[data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }

    .element-container {
        margin-bottom: 0.25rem !important;
    }

    .stMarkdown, .stSubheader, .stHeader {
        margin-bottom: 0.15rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎸 Amazon Musical Instrument Review Sentiment Analysis")
st.markdown("""
This web app analyzes customer reviews and predicts sentiment (Positive, Neutral, or Negative).
It compares multiple machine learning models and provides detailed insights.
""")

# Initialize session state for user reviews
if 'user_reviews' not in st.session_state:
    st.session_state.user_reviews = {'Positive': [], 'Negative': [], 'Neutral': []}

# ============================================
# FILE UPLOAD SECTION
# ============================================
st.sidebar.header("📁 Upload Your Own Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with reviews", type=["csv"])
use_uploaded = uploaded_file is not None

if use_uploaded:
    st.sidebar.success("Using uploaded dataset!")
else:
    st.sidebar.info("Using default Amazon reviews dataset.")

# ============================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================
@st.cache_resource
def load_and_prepare_data(uploaded_file=None):
    """Load and preprocess the dataset"""
    try:
        if uploaded_file is not None:
            dataset = pd.read_csv(uploaded_file)

            if 'reviewText' in dataset.columns and 'summary' in dataset.columns and 'overall' in dataset.columns:
                pass
            else:
                text_cols = [col for col in dataset.columns if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower()]
                summary_cols = [col for col in dataset.columns if 'summary' in col.lower() or 'title' in col.lower()]
                rating_cols = [col for col in dataset.columns if 'rating' in col.lower() or 'overall' in col.lower() or 'score' in col.lower()]

                if text_cols:
                    dataset = dataset.rename(columns={text_cols[0]: 'reviewText'})
                if summary_cols:
                    dataset = dataset.rename(columns={summary_cols[0]: 'summary'})
                if rating_cols:
                    dataset = dataset.rename(columns={rating_cols[0]: 'overall'})

                if not all(col in dataset.columns for col in ['reviewText', 'summary', 'overall']):
                    st.error("Uploaded CSV must contain columns for review text, summary, and rating (e.g., 'reviewText', 'summary', 'overall').")
                    return None
        else:
            dataset = pd.read_csv("Instruments_Reviews.csv")

        dataset["reviewText"] = dataset["reviewText"].fillna("")
        dataset["summary"] = dataset["summary"].fillna("")

        dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]
        dataset.drop(columns=["reviewText", "summary"], inplace=True)

        def label_sentiment(rows):
            if rows["overall"] > 3.0:
                return "Positive"
            elif rows["overall"] < 3.0:
                return "Negative"
            else:
                return "Neutral"

        dataset["sentiment"] = dataset.apply(label_sentiment, axis=1)

        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

CACHE_PATH = Path("model_cache.joblib")

def save_model_cache(tfidf, encoder, trained_models, model_results):
    dump({
        "tfidf": tfidf,
        "encoder": encoder,
        "trained_models": trained_models,
        "model_results": model_results
    }, CACHE_PATH)

def load_model_cache():
    if CACHE_PATH.exists():
        return load(CACHE_PATH)
    return None

@st.cache_resource
def preprocess_reviews(texts):
    """Clean and process review text without punkt dependency"""
    stopwords_set = set(stopwords.words("english")) - {"not"}
    lemmatizer = WordNetLemmatizer()

    processed_texts = []

    for text in texts:
        text = str(text).lower()

        punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        text = text.translate(punc)

        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\n+', ' ', text)

        # Regex-based tokenization to avoid NLTK punkt / punkt_tab dependency
        tokens = re.findall(r"\b[a-zA-Z]+\b", text)

        processed = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stopwords_set
        ]

        processed_texts.append(" ".join(processed))

    return processed_texts

@st.cache_resource
def train_models(_X_train, _X_test, _y_train, _y_test, svm_kernel='linear'):
    """Train multiple models and return results"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': BernoulliNB(),
        'Support Vector Machine': SVC(random_state=42, kernel=svm_kernel, probability=True)
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(_X_train, _y_train)
        y_pred = model.predict(_X_test)
        accuracy = accuracy_score(_y_test, y_pred)

        trained_models[name] = model
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(_y_test, y_pred),
            'classification_report': classification_report(_y_test, y_pred, output_dict=True)
        }

    return trained_models, results

# Load data
dataset = load_and_prepare_data(uploaded_file if use_uploaded else None)

if dataset is not None and len(dataset) > 0:
    st.sidebar.header("⚡ Speed Mode")
    fast_mode = st.sidebar.checkbox("Fast mode (faster training, lower accuracy)", value=True)

    if fast_mode:
        dataset = dataset.sample(frac=0.5, random_state=42).reset_index(drop=True)
        max_features = 2000
        ngram_range = (1, 2)
        svm_kernel = 'linear'
    else:
        max_features = 5000
        ngram_range = (2, 2)
        svm_kernel = 'rbf'

    st.sidebar.write(f"Training on {len(dataset)} rows using TF-IDF max_features={max_features}")

    cache_data = load_model_cache()
    if cache_data is not None and cache_data.get('tfidf') is not None and cache_data.get('trained_models') is not None and cache_data.get('model_results') is not None:
        tfidf = cache_data['tfidf']
        encoder = cache_data['encoder']
        trained_models = cache_data['trained_models']
        model_results = cache_data['model_results']
        st.sidebar.success('Loaded models from cache.')
    else:
        dataset["reviews"] = preprocess_reviews(dataset["reviews"])

        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X = tfidf.fit_transform(dataset["reviews"])

        encoder = LabelEncoder()
        y = encoder.fit_transform(dataset["sentiment"])

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.25, random_state=42
        )

        trained_models, model_results = train_models(X_train, X_test, y_train, y_test, svm_kernel=svm_kernel)

        save_model_cache(tfidf, encoder, trained_models, model_results)

    # ============================================
    # 2. SIDEBAR - INPUT & SETTINGS
    # ============================================
    st.sidebar.header("📝 Input Review")
    user_review = st.sidebar.text_area(
        "Enter your review:",
        placeholder="Type a customer review here...",
        height=120
    )

    predict_button = st.sidebar.button("🔮 Predict Sentiment", width='stretch')

    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose model for prediction:",
        list(trained_models.keys())
    )

    # ============================================
    # 3. SENTIMENT PREDICTION (MAIN SECTION)
    # ============================================
    if predict_button and user_review.strip():
        with st.spinner("🔄 Processing review..."):
            processed_review = preprocess_reviews([user_review])[0]
            feature_vector = tfidf.transform([processed_review])

            model = trained_models[selected_model]
            prediction = model.predict(feature_vector)[0]
            prediction_proba = model.predict_proba(feature_vector)[0]

            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment_label = label_map[prediction]
            confidence = max(prediction_proba) * 100

            emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", sentiment_label)
            with col2:
                st.metric("Confidence", f"{confidence:.2f}%")
            with col3:
                st.metric("Model Used", selected_model)

            st.subheader("Confidence Score Breakdown")
            confidence_df = pd.DataFrame({
                'Sentiment': ['Negative', 'Neutral', 'Positive'],
                'Confidence (%)': prediction_proba * 100
            })

            fig, ax = plt.subplots(figsize=(6, 2))
            colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
            ax.barh(confidence_df['Sentiment'], confidence_df['Confidence (%)'], color=colors)
            ax.set_xlabel('Confidence Score (%)')
            ax.set_xlim(0, 100)
            for i, v in enumerate(confidence_df['Confidence (%)']):
                ax.text(v + 1, i, f'{v:.1f}%', va='center')
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("📝 Your Review Analysis")
            st.write(f"**Review Text:** {user_review}")
            st.write(f"**Predicted Sentiment:** {sentiment_label} {emoji_map[sentiment_label]}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")

            st.session_state.user_reviews[sentiment_label].append(user_review)

    # ============================================
    # BATCH ANALYSIS FOR UPLOADED DATASET
    # ============================================
    if use_uploaded and st.sidebar.button("🔍 Analyze Entire Uploaded Dataset", width='stretch'):
        with st.spinner("🔄 Analyzing all reviews..."):
            all_reviews = preprocess_reviews(dataset["reviews"])
            feature_vectors = tfidf.transform(all_reviews)

            model = trained_models[selected_model]
            predictions = model.predict(feature_vectors)
            prediction_probas = model.predict_proba(feature_vectors)

            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            predicted_sentiments = [label_map[pred] for pred in predictions]
            confidences = [max(proba) * 100 for proba in prediction_probas]

            results_df = dataset.copy()
            results_df["Predicted_Sentiment"] = predicted_sentiments
            results_df["Confidence"] = confidences

            st.success(f"✅ Analyzed {len(results_df)} reviews!")

            st.subheader("📊 Batch Analysis Summary")
            summary = results_df["Predicted_Sentiment"].value_counts()
            st.write(summary)

            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                width='stretch'
            )

            st.subheader("🔍 Sample Results")
            st.dataframe(results_df.head(10), width='stretch')

    # ============================================
    # 4. MODEL COMPARISON SECTION
    # ============================================
    st.markdown("---")
    st.header("📊 Model Performance Comparison")

    col1, col2, col3 = st.columns(3)

    for idx, (model_name, results) in enumerate(model_results.items()):
        with [col1, col2, col3][idx]:
            accuracy = results['accuracy'] * 100
            st.metric(model_name, f"{accuracy:.2f}%")

    st.subheader("Model Accuracy Comparison")
    accuracy_data = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy (%)': [results['accuracy'] * 100 for results in model_results.values()]
    })

    fig, ax = plt.subplots(figsize=(8, 3))
    colors_models = ['#3498db', '#e74c3c', '#2ecc71']
    ax.bar(accuracy_data['Model'], accuracy_data['Accuracy (%)'], color=colors_models)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% Baseline')
    for i, v in enumerate(accuracy_data['Accuracy (%)']):
        ax.text(i, v + 1, f'{v:.2f}%', ha='center', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # ============================================
    # 5. CONFUSION MATRIX & CLASSIFICATION REPORT
    # ============================================
    st.markdown("---")
    st.header("🎯 Detailed Model Analysis")

    analysis_model = st.selectbox(
        "Select model for detailed analysis:",
        list(model_results.keys()),
        key="analysis_model"
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"{analysis_model} - Confusion Matrix")
        cm = model_results[analysis_model]['confusion_matrix']

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            cbar=True,
            ax=ax
        )
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader(f"{analysis_model} - Classification Report")
        report = model_results[analysis_model]['classification_report']
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width='stretch')

    # ============================================
    # 6. DATASET INSIGHTS
    # ============================================
    st.header("📈 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", len(dataset))
    with col2:
        st.metric("Avg Review Length", f"{dataset['reviews'].str.len().mean():.0f} chars")
    with col3:
        st.metric("Unique Sentiments", len(dataset['sentiment'].unique()))

    st.subheader("Sentiment Distribution in Dataset")
    sentiment_counts = dataset['sentiment'].value_counts()

    fig, ax = plt.subplots(figsize=(7, 2.8))
    colors_sentiment = ['#2ecc71', '#ffd93d', '#e74c3c']

    sentiment_counts.plot(kind='bar', ax=ax, color=colors_sentiment[:len(sentiment_counts)])

    ax.set_ylabel('Count')
    ax.set_xlabel('Sentiment')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    max_count = max(sentiment_counts.max(), 1)
    offset = max(0.05 * max_count, 0.2)

    ax.set_ylim(0, max_count + offset * 3)

    for i, v in enumerate(sentiment_counts):
        ax.text(i, v + offset, str(v), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("☁️ Word Clouds")

    col1, col2, col3 = st.columns(3)

    sentiments = ['Positive', 'Neutral', 'Negative']
    colors = ['Greens', 'Blues', 'Reds']

    for i, sentiment in enumerate(sentiments):
        with [col1, col2, col3][i]:
            st.markdown(f"**{sentiment} Reviews**")
            dataset_reviews = " ".join(dataset[dataset['sentiment'] == sentiment]['reviews'])
            user_reviews = " ".join(st.session_state.user_reviews.get(sentiment, []))
            all_reviews = (dataset_reviews + " " + user_reviews).strip()

            if all_reviews:
                wordcloud = WordCloud(
                    width=350,
                    height=180,
                    background_color='white',
                    colormap=colors[i]
                ).generate(all_reviews)

                fig, ax = plt.subplots(figsize=(3.8, 2.1))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0.2)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info(f"No {sentiment.lower()} reviews to display.")

    st.subheader("Sample Reviews from Dataset")

    tab1, tab2, tab3 = st.tabs(["Positive", "Neutral", "Negative"])

    with tab1:
        positive_samples = dataset[dataset['sentiment'] == 'Positive']['reviews'].head(3)
        for idx, review in enumerate(positive_samples, 1):
            st.write(f"**Review {idx}:**")
            st.write(review[:200] + "..." if len(review) > 200 else review)

    with tab2:
        neutral_samples = dataset[dataset['sentiment'] == 'Neutral']['reviews'].head(3)
        for idx, review in enumerate(neutral_samples, 1):
            st.write(f"**Review {idx}:**")
            st.write(review[:200] + "..." if len(review) > 200 else review)

    with tab3:
        negative_samples = dataset[dataset['sentiment'] == 'Negative']['reviews'].head(3)
        for idx, review in enumerate(negative_samples, 1):
            st.write(f"**Review {idx}:**")
            st.write(review[:200] + "..." if len(review) > 200 else review)

else:
    st.error("Failed to load dataset. Please ensure 'Instruments_Reviews.csv' is in the same directory.")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📚 Amazon Musical Instrument Review Sentiment Analysis</p>
    <p><small>Built with Streamlit | Machine Learning | NLP</small></p>
</div>
""", unsafe_allow_html=True)
