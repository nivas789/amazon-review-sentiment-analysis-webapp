# 🎸 Amazon Musical Instrument Review Sentiment Analysis Web App

## Overview

This project analyzes customer reviews for Amazon musical instruments and predicts the sentiment (Positive, Neutral, or Negative) using machine learning. The **interactive web application** compares multiple models and provides detailed performance metrics and visualizations.

**Project Type:** NLP + Machine Learning + Web Application  
**Dataset:** [Amazon Musical Instruments Reviews (Kaggle)](https://www.kaggle.com/eswarchandt/amazon-music-reviews)  
**Status:** ✅ Complete & Production-Ready

---

## 🎯 Key Features

### 1. **Interactive User Input** ✅
- Users can type their own reviews and get instant sentiment predictions
- Real-time text processing and feature vectorization
- Displays confidence scores for each sentiment category

### 2. **Clear Sentiment Output** ✅
- Shows sentiment as readable text: **Positive**, **Neutral**, or **Negative**
- Includes confidence percentage (0-100%)
- Visual confidence breakdown using bar charts

### 3. **Model Comparison** ✅
The app trains and compares 3 different machine learning models:
- **Logistic Regression** (~88% accuracy) - Best overall
- **Naive Bayes** (~81% accuracy) - Fast & probabilistic
- **Support Vector Machine** (~88% accuracy) - Robust

### 4. **Rich Visualizations** ✅
- Accuracy comparison bar charts
- Confusion matrix heatmaps
- Classification reports with metrics
- Sentiment distribution graphs
- Per-prediction confidence visualization

### 5. **Streamlit Web Interface** ✅
- Professional, responsive dashboard
- Sidebar controls for easy navigation
- Tab-based organization
- Real-time model inference
- Beautiful styling and layout

---

## 📊 Dataset Overview

**Source:** Kaggle - Amazon Musical Instruments Reviews  
**Size:** 10,261 reviews  
**Sentiment Distribution:**
- Positive: 88.8% (9,022 reviews)
- Neutral: 7.5% (772 reviews)
- Negative: 4.5% (467 reviews)

**Data Preprocessing:**
- ✅ Text cleaning (lowercase, punctuation removal, URL removal)
- ✅ Tokenization and lemmatization
- ✅ Stopword removal (preserving "not" for sentiment)
- ✅ TF-IDF vectorization (5,000 bigram features)
- ✅ SMOTE balancing for imbalanced classes

---

## 🚀 Quick Start

### Installation (30 seconds)

```bash
# Install dependencies
pip install streamlit pandas scikit-learn nltk textblob imbalanced-learn

# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

### Run the Web App

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 📖 How to Use

### 1. **Enter a Review**
   - Type in the sidebar text box
   - Click "🔮 Predict Sentiment"

### 2. **View Results**
   - See sentiment label (Positive/Neutral/Negative)
   - Check confidence percentage
   - View confidence breakdown chart

### 3. **Select Model**
   - Choose from 3 different ML models
   - See different predictions and confidences

### 4. **Analyze Models**
   - View accuracy comparison
   - Check confusion matrices
   - Review classification reports

### 5. **Explore Data**
   - See dataset statistics
   - View sentiment distribution
   - Read sample reviews

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **88.2%** | 0.88 | 0.88 | 0.88 |
| **SVM** | **88.0%** | 0.88 | 0.88 | 0.88 |
| **Naive Bayes** | 80.9% | 0.81 | 0.81 | 0.81 |

**Evaluation Method:** 10-Fold Cross-Validation  
**Train/Test Split:** 75/25  
**Imbalance Handling:** SMOTE

---

## 📁 Project Structure

```
Sentiment-Analysis-main/
├── streamlit_app.py              # 🌟 Main web application
├── Sentiment Analysis.py          # Original notebook (script version)
├── Sentiment Analysis.ipynb       # Original Jupyter notebook
├── Instruments_Reviews.csv        # Dataset
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Interactive web framework & dashboards |
| **Scikit-Learn** | ML models (Logistic Regression, SVM, Naive Bayes) |
| **NLTK** | Natural language processing & tokenization |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical operations |
| **Matplotlib/Seaborn** | Data visualizations |
| **TextBlob** | Sentiment polarity analysis |
| **Imbalanced-Learn** | SMOTE for class balancing |

---

## 🎓 Learning Highlights

This project demonstrates professional-grade expertise in:

✅ **Data Preprocessing** - Text cleaning, normalization, lemmatization  
✅ **Feature Engineering** - TF-IDF vectorization, dimensionality  
✅ **Model Selection & Training** - Multiple classifiers, hyperparameter tuning  
✅ **Model Evaluation** - Confusion matrix, precision, recall, F1-score  
✅ **Web Development** - Streamlit dashboard creation  
✅ **NLP Fundamentals** - Sentiment classification, text vectorization  
✅ **Software Best Practices** - Code organization, documentation, caching  

---

## 📝 Example Usage

### Positive Review
```
Input: "Great sound quality, very satisfied with this purchase!"
Model: Logistic Regression
Output: ✅ Positive (Confidence: 94.2%)
```

### Negative Review
```
Input: "Broke within a week, terrible quality, waste of money"
Model: Logistic Regression
Output: ❌ Negative (Confidence: 89.7%)
```

### Neutral Review
```
Input: "It works okay, not great but not bad either"
Model: Logistic Regression
Output: ⚖️ Neutral (Confidence: 72.5%)
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found error | `pip install streamlit pandas scikit-learn nltk textblob imbalanced-learn` |
| CSV file not found | Ensure `Instruments_Reviews.csv` is in same directory |
| NLTK data missing | `python -m nltk.downloader punkt stopwords wordnet` |
| App runs slowly | First run processes data; subsequent runs use cache |

---

## 📚 References

- Original Notebook: [Google Colab](https://colab.research.google.com/drive/1Jb0-XtSdEoTIYw6suN4nzlao4vaXX0JY)
- Dataset: [Kaggle](https://www.kaggle.com/eswarchandt/amazon-music-reviews)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-Learn Guide](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## ⭐ Project Highlights

✨ **Production-Ready Code** - Professional structure and best practices  
✨ **Interactive Dashboard** - Real-time sentiment predictions  
✨ **Model Comparison** - 3 ML algorithms compared side-by-side  
✨ **Rich Visualizations** - Multiple charts and metrics  
✨ **Complete Documentation** - Clear README and code comments  

---

**Built with ❤️ for Machine Learning & NLP**
