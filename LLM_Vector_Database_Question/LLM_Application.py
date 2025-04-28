import os
import time
import pandas as pd
from typing import List, Dict, Any
from google.generativeai import GenerativeModel, configure

# Configure your Google API key
configure(api_key="***********")  # Removed for security reasons

class ReviewLLMProcessor:
    """Process reviews using Google's Gemini model."""

    def __init__(self, model_name="gemini-1.5-pro"):
        """Initialize with Gemini model."""
        self.model = GenerativeModel(model_name)
    
    def generate_text(self, prompt: str, max_tokens=600, temperature=0.7) -> str:
        """Generate text completion."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating text: {str(e)}"

    def generate_category_summary(self, vector_db, category):
        """Generate a summary of product performance for a specific category."""
        # Filter reviews
        category_reviews = vector_db.df[vector_db.df['category'] == category]
        
        if len(category_reviews) == 0:
            return f"No reviews found for category: {category}"
        
        # Get statistics
        avg_rating = category_reviews['rating'].mean()
        sentiment_counts = category_reviews['sentiment'].value_counts()
        
        # Sample reviews
        sample_reviews = []
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_reviews = category_reviews[category_reviews['sentiment'] == sentiment]
            if len(sentiment_reviews) > 0:
                sample_reviews.append(sentiment_reviews.sample(min(3, len(sentiment_reviews))))
        
        sample_reviews = pd.concat(sample_reviews).reset_index(drop=True)

        # Prompt Engineering
        prompt = f"""
        Summarize customer reviews for the {category} category.

        - Average Rating: {avg_rating:.2f}/5
        - Total Reviews: {len(category_reviews)}
        - Sentiment Distribution: {sentiment_counts.to_dict()}

        Sample Reviews:
        {sample_reviews[['product', 'rating', 'sentiment', 'review_text']].to_string(index=False)}

        Instructions:
        1. Highlight common strengths and weaknesses.
        2. Identify standout products.
        3. Mention recurring issues or praised features.
        4. Write a clear, analytical summary in about 250-300 words.
        """

        return self.generate_text(prompt)

    def generate_all_category_summaries(self, vector_db):
        """Generate summaries for all categories."""
        categories = vector_db.df['category'].unique()
        summaries = {}
        
        for category in categories:
            print(f"Generating summary for {category}...")
            summaries[category] = self.generate_category_summary(vector_db, category)
            time.sleep(1)  # Be polite
        
        return summaries

    def answer_question(self, vector_db, question, k=5):
        """Answer a question about products based on reviews."""
        # Search for relevant reviews
        relevant_reviews = vector_db.search(question, k=k)
        
        if not relevant_reviews:
            return "I couldn't find relevant reviews to answer your question."

        # Format reviews
        reviews_text = "\n\n".join([
            f"Product: {review['product']}\nCategory: {review['category']}\nRating: {review['rating']}/5\nReview: {review['review_text']}"
            for review in relevant_reviews
        ])

        # Prepare prompt
        prompt = f"""
        Answer the following customer question using the provided reviews.

        Question: {question}

        Relevant Reviews:
        {reviews_text}

        Instructions:
        1. Provide a direct and concise answer (100-150 words).
        2. Reference specific products if possible.
        3. If the information is insufficient, mention it.
        """

        return self.generate_text(prompt, max_tokens=300)

    def identify_common_issues_and_features(self, vector_db):
        """Identify common praised features and issues across categories."""
        categories = vector_db.df['category'].unique()
        category_insights = {}

        for category in categories:
            category_reviews = vector_db.df[vector_db.df['category'] == category]
            positive_reviews = category_reviews[category_reviews['sentiment'] == 'positive']
            negative_reviews = category_reviews[category_reviews['sentiment'] == 'negative']

            if len(positive_reviews) > 0:
                positive_features = positive_reviews['feature_mentioned'].value_counts().nlargest(5).to_dict()
                positive_attributes = positive_reviews['attribute_mentioned'].value_counts().nlargest(5).to_dict()
            else:
                positive_features = {}
                positive_attributes = {}

            if len(negative_reviews) > 0:
                negative_features = negative_reviews['feature_mentioned'].value_counts().nlargest(5).to_dict()
                negative_attributes = negative_reviews['attribute_mentioned'].value_counts().nlargest(5).to_dict()
            else:
                negative_features = {}
                negative_attributes = {}

            category_insights[category] = {
                'praised_features': positive_features,
                'praised_attributes': positive_attributes,
                'criticized_features': negative_features,
                'criticized_attributes': negative_attributes
            }

        # Prepare insights text
        insights_text = ""
        for category, insights in category_insights.items():
            insights_text += f"\n\n{category.upper()}\n"
            insights_text += f"Praised Features: {insights['praised_features']}\n"
            insights_text += f"Praised Attributes: {insights['praised_attributes']}\n"
            insights_text += f"Criticized Features: {insights['criticized_features']}\n"
            insights_text += f"Criticized Attributes: {insights['criticized_attributes']}\n"

        # Prepare prompt
        prompt = f"""
        Analyze the following customer review insights across different product categories.

        {insights_text}

        Instructions:
        1. Identify cross-category strengths and weaknesses.
        2. Highlight category-specific praised features and common issues.
        3. Suggest improvements for product teams.
        4. Write around 400-500 words.
        """

        return self.generate_text(prompt, max_tokens=800)

def create_qa_system(vector_db, llm_processor):
    """Create a Q&A system for products."""
    def qa_system(question):
        return llm_processor.answer_question(vector_db, question)
    return qa_system

# Example usage
if __name__ == "__main__":
    # Load your vector database
    vector_db = ReviewVectorDB.load("review_vector_db")  # Assuming you have this class ready

    # Initialize the LLM processor
    llm_processor = ReviewLLMProcessor()

    # Create Q&A system
    qa = create_qa_system(vector_db, llm_processor)

    # Example questions
    questions = [
        "Which smartphone has the best battery life?",
        "What are common issues with laptops?",
        "Are there any smart home devices that are difficult to set up?"
    ]

    for question in questions:
        print(f"Q: {question}")
        print(f"A: {qa(question)}")
        print()
