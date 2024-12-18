# task_1_eda.py
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load historical stock data
aapl_data = pd.read_csv('Data/yfinance_data/AAPL_historical_data.csv')
amzn_data = pd.read_csv('Data/yfinance_data/AMZN_historical_data.csv')
goog_data = pd.read_csv('Data/yfinance_data/GOOG_historical_data.csv')
meta_data = pd.read_csv('Data/yfinance_data/META_historical_data.csv')
msft_data = pd.read_csv('Data/yfinance_data/MSFT_historical_data.csv')
nvda_data = pd.read_csv('Data/yfinance_data/NVDA_historical_data.csv')
tsla_data = pd.read_csv('Data/yfinance_data/TSLA_historical_data.csv')

# Load analyst ratings data
analyst_ratings_data = pd.read_csv('Data/raw_analyst_ratings.csv')

# Data preview
print("AAPL Data Preview:\n", aapl_data.head())
print("Analyst Ratings Data Preview:\n", analyst_ratings_data.head())

# Add headline length column to analyst data
analyst_ratings_data['headline_length'] = analyst_ratings_data['headline'].apply(len)


# Descriptive statistics for headline length
headline_length_stats = analyst_ratings_data['headline_length'].describe()
print("Headline Length Stats:\n", headline_length_stats)

# Count the number of articles per publisher
articles_per_publisher = analyst_ratings_data.groupby('publisher').size()
print("Articles per Publisher:\n", articles_per_publisher)

# Convert publication date to datetime format
#analyst_ratings_data['date'] = pd.to_datetime(analyst_ratings_data['date'])
# Use specific format to avoid errors if the format is consistent
analyst_ratings_data['date'] = pd.to_datetime(analyst_ratings_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


# Extract year and month
analyst_ratings_data['year'] = analyst_ratings_data['date'].dt.year
analyst_ratings_data['month'] = analyst_ratings_data['date'].dt.month

# Articles per year and month
articles_per_year = analyst_ratings_data.groupby('year').size()
articles_per_month = analyst_ratings_data.groupby('month').size()

print("Articles Per Year:\n", articles_per_year)
print("Articles Per Month:\n", articles_per_month)

# Sentiment analysis on the headlines
analyst_ratings_data['sentiment'] = analyst_ratings_data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Descriptive statistics for sentiment
sentiment_stats = analyst_ratings_data['sentiment'].describe()
print("Sentiment Stats:\n", sentiment_stats)

# TF-IDF vectorization for topic modeling
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(analyst_ratings_data['headline'])

# LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Print top words for each topic
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic {topic_idx}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])

# Plot publication frequency over time
plt.figure(figsize=(10, 6))
analyst_ratings_data.groupby('date').size().plot()
plt.title('Article Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()

# Top 10 publishers
top_publishers = articles_per_publisher.sort_values(ascending=False).head(10)
print("Top 10 Publishers:\n", top_publishers)
