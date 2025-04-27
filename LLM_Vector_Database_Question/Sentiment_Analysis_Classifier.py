import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class SentimentClassifier:
    """Sentiment classifier for product reviews."""
    
    def __init__(self, model_type="sklearn"):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_type: Type of model to use ("sklearn" or "transformer")
        """
        self.model_type = model_type
        self.model = None
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def train(self, df):
        """Train the sentiment classifier."""
        # Split data
        X = df['review_text'].tolist()
        y = df['sentiment'].map(self.label_map).tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.model_type == "sklearn":
            # Create and train pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000)),
                ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
            ])
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Print evaluation metrics
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=self.label_map.keys()))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.label_map.keys(), 
                        yticklabels=self.label_map.keys())
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
            
            return accuracy_score(y_test, y_pred)
        
        elif self.model_type == "transformer":
            # Load pre-trained model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=3
            )
            
            # Create dataset
            class ReviewDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length=128):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    label = self.labels[idx]
                    
                    encoding = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    return {
                        'input_ids': encoding['input_ids'].flatten(),
                        'attention_mask': encoding['attention_mask'].flatten(),
                        'labels': torch.tensor(label, dtype=torch.long)
                    }
            
            # Create datasets
            train_dataset = ReviewDataset(X_train, y_train, self.tokenizer)
            test_dataset = ReviewDataset(X_test, y_test, self.tokenizer)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16)
            
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Training
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
            num_epochs = 3
            
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    loss.backward()
                    optimizer.step()
                
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluation
            self.model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            
            # Print evaluation metrics
            print("Classification Report:")
            print(classification_report(all_labels, all_preds, 
                                       target_names=self.label_map.keys()))
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.label_map.keys(), 
                        yticklabels=self.label_map.keys())
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()
            
            return accuracy_score(all_labels, all_preds)
    
    def predict(self, texts):
        """Predict sentiment for the given texts."""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type == "sklearn":
            # Make predictions
            predictions = self.model.predict(texts)
            
            # Convert to labels
            return [self.reverse_label_map[pred] for pred in predictions]
        
        elif self.model_type == "transformer":
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            
            # Tokenize texts
            encoded_texts = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                input_ids = encoded_texts['input_ids'].to(device)
                attention_mask = encoded_texts['attention_mask'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                _, preds = torch.max(outputs.logits, dim=1)
            
            # Convert to labels
            return [self.reverse_label_map[pred.item()] for pred in preds]
    
    def evaluate_against_labels(self, df):
        """Evaluate classifier against provided sentiment labels."""
        if not self.model:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        predictions = self.predict(df['review_text'].tolist())
        
        # Compare with actual labels
        actual = df['sentiment'].tolist()
        
        # Calculate metrics
        accuracy = sum(1 for p, a in zip(predictions, actual) if p == a) / len(predictions)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(actual, predictions))
        
        # Confusion matrix
        cm = confusion_matrix(
            [self.label_map[a] for a in actual], 
            [self.label_map[p] for p in predictions]
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_map.keys(), 
                    yticklabels=self.label_map.keys())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        return accuracy