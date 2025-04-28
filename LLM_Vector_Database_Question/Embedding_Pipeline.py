import pandas as pd
import numpy as np
import json
from torch import * 
from sentence_transformers import SentenceTransformer # BERT Transformer for generating embeddings
from sklearn.metrics.pairwise import cosine_similarity # for cosine similarity
import faiss # Facebook AI Similarity Search
import pickle #save/load embeddings and index


def load_data(file_path):
    """Load the product review data from CSV"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file format. Please provide CSV")

def preprocess_reviews(df):
    """Preprocess the review data."""
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a combined text field for embedding
    df['combined_text'] = df['review_text'] + " Product: " + df['product'] + " Category: " + df['category'] + \
                         " Feature: " + df['feature_mentioned'] + " Attribute: " + df['attribute_mentioned']
    
    # Handle missing values
    df = df.fillna('')
    
    return df

def generate_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Generate embeddings for the provided texts using a Sentence Transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    """Build a FAISS index for fast similarity search."""
    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    
    # Create the index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings)
    
    return index

class ReviewVectorDB:
    """Vector database for product reviews."""
    
    def __init__(self, df=None, embeddings=None, index=None):
        self.df = df
        self.embeddings = embeddings
        self.index = index
        self.model = None
    
    def initialize(self, file_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector database from a file."""
        # Load and preprocess data
        df = load_data(file_path)
        self.df = preprocess_reviews(df)
        # Load model
        self.model = SentenceTransformer(model_name)
        # Generate embeddings
        self.embeddings = generate_embeddings(self.df['combined_text'].tolist(), model_name)
        # Build index
        self.index = build_faiss_index(self.embeddings)
        return self
    
    def save(self, path_prefix):
        """Save the vector database to disk."""
        # Save dataframe
        self.df.to_pickle(f"{path_prefix}_df.pkl")
        # Save embeddings
        with open(f"{path_prefix}_embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)   
        # Save index
        faiss.write_index(self.index, f"{path_prefix}_index.faiss")
    
    @classmethod
    def load(cls, path_prefix, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Load the vector database from disk."""
        # Load dataframe
        df = pd.read_pickle(f"{path_prefix}_df.pkl")    
        # Load embeddings
        with open(f"{path_prefix}_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        # Load index
        index = faiss.read_index(f"{path_prefix}_index.faiss")
        # Create instance
        instance = cls(df, embeddings, index)
        instance.model = SentenceTransformer(model_name)
        return instance
    
    def search(self, query, k=5):
        """Search for similar reviews."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)   
        # Search
        D, I = self.index.search(query_embedding, k)
        # Return results
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.df):  # Ensure index is valid
                result = self.df.iloc[idx].to_dict()
                #result['similarity'] = float(distance)
                results.append(result)
        
        return results
    
    def filter_search(self, query, filters=None, k=5):
        """Search with filters (post-filtering approach)."""
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search more results than needed to allow for filtering
        D, I = self.index.search(query_embedding, k*5)
        
        # Filter results
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.df):  # Ensure index is valid
                result = self.df.iloc[idx].to_dict()
                
                # Apply filters
                if filters:
                    match = True
                    for key, value in filters.items():
                        if key in result and result[key] != value:
                            match = False
                            break
                    
                    if not match:
                        continue
                
                result['similarity'] = float(distance)
                results.append(result)
                
                if len(results) >= k:
                    break
        
        return results[:k]

# Example usage
if __name__ == "__main__":
    # Initialize and save
    vector_db = ReviewVectorDB().initialize("/content/product_reviews.csv")
    vector_db.save("review_vector_db")
    
    # Load and search
    vector_db = ReviewVectorDB.load("review_vector_db")
    results = vector_db.search("battery life issues", k=3)
    print(results)