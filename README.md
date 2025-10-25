# 📰 Fake News Classification System using Web Mining and LSTM

## 📘 Project Overview
The **Fake News Classification System** was developed to explore how **Web Mining** and **Machine Learning** can be applied to combat misinformation on the internet.  
This project detects and classifies fake news in **real time** by scraping web content, preprocessing the collected data, and using a **Long Short-Term Memory (LSTM)** deep learning model to predict whether an article is *real* or *fake*.

It demonstrates a complete end-to-end pipeline — from **data collection**, **cleaning**, and **exploration**, to **deep learning-based classification** — providing insights into linguistic patterns that differentiate fake from real news.

---

## 🧩 Project Structure
|-- Fake_news_preprocessing.ipynb # Data cleaning, text normalization, tokenization  
|-- fake news Analysis.ipynb # Exploratory data analysis and visualizations  
|-- fake_news_detection_LSTM.ipynb # LSTM model training and evaluation  
|-- Fake.csv # Fake news dataset  
|-- True.csv # True/real news dataset  
|-- fake_news.csv # Combined and processed dataset  
|-- README.md # Project documentation  


---

## 🌐 Web Mining Component
The system uses **web scraping** to collect news articles from multiple online sources.  
- Web pages are scraped using Python libraries such as **BeautifulSoup** and **Requests**.  
- Extracted content includes headlines, article text, and publication metadata.  
- Collected data is merged with the offline datasets (`Fake.csv`, `True.csv`) for enhanced model training and testing.  
- The aim is to demonstrate **real-time classification** of scraped data using the trained model.

---

## 🧠 Model Description

### Model Type
The core classification model is built using a **Recurrent Neural Network (RNN)** with **LSTM** layers.  
LSTM networks are ideal for this problem as they capture long-term dependencies in text data.

### Processing & Model Pipeline
1. **Data Preprocessing**
   - Text lowercasing  
   - Removal of punctuation, stopwords, and special characters  
   - Tokenization and sequence padding using `Tokenizer` from Keras  

2. **Exploratory Data Analysis**
   - Visualization of word frequency, subject category distribution, and word clouds  
   - Balanced dataset verification  

3. **Model Architecture**
   - Embedding layer for text representation  
   - LSTM layer for sequence learning  
   - Fully connected Dense layers  
   - Sigmoid output for binary classification  

4. **Training Configuration**
   - Optimizer: `Adam`  
   - Loss: `Binary Crossentropy`  
   - Epochs: 5–10 (configurable)  
   - Evaluation metrics: Accuracy, Precision, Recall, F1-score  

5. **Evaluation**
   - Model tested on unseen data  
   - Confusion matrix and performance metrics plotted  

---

## 📈 Results

| Metric | Result |
|:--------|:--------|
| **Accuracy** | ~98% |
| **Precision** | 0.97 |
| **Recall** | 0.98 |
| **F1-Score** | 0.975 |

The LSTM model achieved high accuracy in distinguishing real and fake news articles.  
Visualization outputs such as confusion matrices and word distributions confirm the model’s reliability.

---

## 📊 Dataset Details

The dataset consists of:
- **Fake.csv:** Fake or misleading news articles  
- **True.csv:** Genuine and verified news articles  

Each dataset contains:
- `title`: Headline of the article  
- `text`: Article content  
- `subject`: Category/topic  
- `date`: Publication date  

After preprocessing, both datasets are merged into a single file — **`fake_news.csv`** — used for model training.

Dataset source: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

## 🚀 How to Run the Project

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/fake-news-classification-system.git
   cd fake-news-classification-system
   ```
2. **Install required dependencies**
    ```bash
    pip install -r requirements.txt
    
3. **Run the notebooks in the following order**

    - Fake_news_preprocessing.ipynb

    - fake news Analysis.ipynb

    - fake_news_detection_LSTM.ipynb
