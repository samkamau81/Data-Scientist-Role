# Import libraries
import matplotlib.pyplot as plt
import plotly.express as px

# Load your data
df = pd.read_csv('/content/product_reviews.csv')

# If your sentiment predictions are not yet added, use your SentimentClassifier to predict:
# sentiment_model = SentimentClassifier(model_type="your_best_model")
# sentiment_model.train(df)
# df['predicted_sentiment'] = sentiment_model.predict(df['review_text'].tolist())

# ----- Prepare for Visualization -----

# Convert your 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])  # change 'review_date' if your date column has a different name

# Map sentiments to numerical scale for easier plotting if needed
sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

# Group by month
df['month'] = df['date'].dt.to_period('M').astype(str)

# Group by category
if 'category' not in df.columns:
    raise ValueError("Your dataframe must have a 'product_category' column.")

# ----- Visualization Part -----

# 1. Sentiment Trends Over Time
plt.figure(figsize=(14,6))
monthly_sentiment = df.groupby('month')['sentiment_score'].mean()
sns.lineplot(x=monthly_sentiment.index, y=monthly_sentiment.values, marker='o')
plt.xticks(rotation=45)
plt.title('Average Sentiment Over Time')
plt.xlabel('Month')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.show()

# 2. Sentiment Distribution Across Categories
plt.figure(figsize=(14,8))
category_sentiment = df.groupby('category')['sentiment_score'].mean().sort_values()
sns.barplot(x=category_sentiment.values, y=category_sentiment.index, palette='coolwarm')
plt.title('Average Sentiment Score Across Product Categories')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Product Category')
plt.grid(True)
plt.show()

# 3. Number of Reviews per Sentiment per Category (Stacked bar)
sentiment_counts = df.groupby(['category', 'sentiment']).size().unstack().fillna(0)

sentiment_counts.plot(kind='bar', stacked=True, figsize=(16,8), colormap='viridis')
plt.title('Sentiment Distribution per Product Category')
plt.xlabel('Product Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# 4. Interactive Dashboard with Plotly (Optional, very beautiful)
fig = px.scatter(
    df, 
    x='date', 
    y='sentiment_score', 
    color='sentiment',
    hover_data=['category', 'review_text'],
    title="Sentiment Trends Over Time (Interactive)"
)
fig.show()
