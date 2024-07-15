import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set your OpenAI API key here
openai.api_key = '' #paste your key

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to identify patterns using clustering and classification
# Function to identify patterns using clustering and classification
def identify_patterns(df):
    # Exclude non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Clustering using K-Means
    kmeans = KMeans(n_clusters=3)
    df['Cluster'] = kmeans.fit_predict(df[numeric_columns])

    # Visualize clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[numeric_columns])
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Cluster'])
    plt.title('K-Means Clustering')
    plt.savefig('kmeans_cluster.png')  # Save the plot
    plt.show()

    # Classification using Logistic Regression
    if 'target_column' in df.columns:
        X = df.drop('target_column', axis=1)
        y = df['target_column']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        print(classification_report(y_test, y_pred))
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')  # Save the plot
    plt.show()

    # Pair plot
    sns.pairplot(df)
    plt.title('Pair Plot')
    plt.savefig('pair_plot.png')  # Save the plot
    plt.show()

    # Create plots for each numeric column
    for column in numeric_columns:
        plt.figure(figsize=(10, 6))
        if df[column].nunique() > 10:
            sns.histplot(df[column], kde=True, color='orange')
        else:
            sns.countplot(x=column, data=df)
        plt.title(f'Plot for {column}')
        plt.savefig(f'{column}_plot.png')  # Save the plot
        plt.show()


# Function to generate insights using GPT-3
def generate_insights_nlg(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Function to summarize text using BERT
def summarize_text(text):
    summarizer = pipeline("summarization")
    summarized_text = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summarized_text[0]['summary_text']

# Function to analyze sentiment using BERT
def analyze_sentiment(text):
    nlp_sentiment = pipeline("sentiment-analysis")
    sentiment_result = nlp_sentiment(text)[0]
    return sentiment_result

# Function to analyze sentiment using NLTK
def analyze_sentiment_nltk(text):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def main():
    file_path = 'preprocessed_data.csv'
    df = load_data(file_path)
    identify_patterns(df)

    # Example usage of NLG, text summarization, and sentiment analysis
    sample_text = "This is a sample text for generating insights."
    generated_insight = generate_insights_nlg(sample_text)
    summarized_text = summarize_text(sample_text)
    sentiment_analysis_bert = analyze_sentiment(sample_text)
    sentiment_analysis_nltk = analyze_sentiment_nltk(sample_text)

if __name__ == "__main__":
    main()
