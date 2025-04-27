import openai
from openai import OpenAI
import os
import time
from typing import List, Dict, Any

class ReviewLLMProcessor:
    """Process reviews using LLMs."""
    
    def __init__(self, api_key=None):
        """Initialize with OpenAI API key."""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY from environment
    
    def generate_category_summary(self, vector_db, category):
        """Generate a summary of product performance for a specific category."""
        # Filter reviews for the category
        category_reviews = vector_db.df[vector_db.df['category'] == category]
        
        if len(category_reviews) == 0:
            return f"No reviews found for category: {category}"
        
        # Get statistics
        avg_rating = category_reviews['rating'].mean()
        sentiment_counts = category_reviews['sentiment'].value_counts()
        
        # Sample reviews (positive, neutral, negative)
        sample_reviews = []
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_reviews = category_reviews[category_reviews['sentiment'] == sentiment]
            if len(sentiment_reviews) > 0:
                sample_reviews.append(sentiment_reviews.sample(min(3, len(sentiment_reviews))))
        
        sample_reviews = pd.concat(sample_reviews).reset_index(drop=True)
        
        # Prepare prompt
        prompt = f"""
        Generate a concise summary of product performance in the {category} category based on the following information:
        
        - Average Rating: {avg_rating:.2f}/5
        - Total Reviews: {len(category_reviews)}
        - Sentiment Distribution: {sentiment_counts.to_dict()}
        
        Sample Reviews:
        {sample_reviews[['product', 'rating', 'sentiment', 'review_text']].to_string()}
        
        Your summary should:
        1. Highlight common strengths and weaknesses
        2. Identify standout products
        3. Note any recurring issues or praised features
        4. Be about 250-300 words
        """
        
        # Generate summary
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a product analyst providing insights based on customer reviews."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_all_category_summaries(self, vector_db):
        """Generate summaries for all product categories."""
        categories = vector_db.df['category'].unique()
        summaries = {}
        
        for category in categories:
            print(f"Generating summary for {category}...")
            summaries[category] = self.generate_category_summary(vector_db, category)
            time.sleep(1)  # Rate limiting
        
        return summaries
    
    def answer_question(self, vector_db, question, k=5):
        """Answer a specific question about products using review data."""
        # Search for relevant reviews
        relevant_reviews = vector_db.search(question, k=k)
        
        if not relevant_reviews:
            return "I couldn't find relevant reviews to answer your question."
        
        # Format reviews for the prompt
        reviews_text = "\n\n".join([
            f"Product: {review['product']}\nCategory: {review['category']}\n" +
            f"Rating: {review['rating']}/5\nReview: {review['review_text']}"
            for review in relevant_reviews
        ])
        
        # Prepare prompt
        prompt = f"""
        Based on the following customer reviews, please answer this question:
        
        Question: {question}
        
        Relevant Reviews:
        {reviews_text}
        
        Your answer should:
        1. Be direct and concise (100-150 words)
        2. Reference specific products and reviews where relevant
        3. Mention if there's insufficient information to fully answer the question
        """
        
        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing product information based on customer reviews."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def identify_common_issues_and_features(self, vector_db):
        """Identify common issues and praised features across product categories."""
        # Get statistics by category
        categories = vector_db.df['category'].unique()
        category_insights = {}
        
        for category in categories:
            # Filter reviews for the category
            category_reviews = vector_db.df[vector_db.df['category'] == category]
            
            # Get positive and negative reviews
            positive_reviews = category_reviews[category_reviews['sentiment'] == 'positive']
            negative_reviews = category_reviews[category_reviews['sentiment'] == 'negative']
            
            # Get common features and attributes mentioned
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
            
            # Store insights
            category_insights[category] = {
                'praised_features': positive_features,
                'praised_attributes': positive_attributes,
                'criticized_features': negative_features,
                'criticized_attributes': negative_attributes
            }
        
        # Format insights for the prompt
        insights_text = ""
        for category, insights in category_insights.items():
            insights_text += f"\n\n{category.upper()}\n"
            insights_text += f"Praised Features: {insights['praised_features']}\n"
            insights_text += f"Praised Attributes: {insights['praised_attributes']}\n"
            insights_text += f"Criticized Features: {insights['criticized_features']}\n"
            insights_text += f"Criticized Attributes: {insights['criticized_attributes']}\n"
        
        # Prepare prompt
        prompt = f"""
        Based on the following statistics about features and attributes mentioned in customer reviews,
        please identify common issues and praised features across product categories.
        
        {insights_text}
        
        Your analysis should:
        1. Identify cross-category strengths and weaknesses
        2. Highlight category-specific issues and praised features
        3. Suggest potential areas for improvement across product lines
        4. Be about 400-500 words in total
        """
        
        # Generate analysis
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a product analyst providing insights based on customer reviews."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
        

def create_qa_system(vector_db, llm_processor):
    """
    Create a Q&A system that can answer specific questions about products.
    
    Args:
        vector_db: Initialized vector database
        llm_processor: LLM processor for generating answers
    
    Returns:
        function: A function that takes a question and returns an answer
    """
    def qa_system(question):
        return llm_processor.answer_question(vector_db, question)
    
    return qa_system

# Example usage
if __name__ == "__main__":
    # Load vector database
    vector_db = ReviewVectorDB.load("review_vector_db")
    
    # Initialize LLM processor
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