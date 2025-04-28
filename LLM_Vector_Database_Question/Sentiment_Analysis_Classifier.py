# ================== Imports ==================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ================== Load Data ==================
# Load your product reviews
df = pd.read_csv('/content/product_reviews.csv')

# Check your columns
print(df.columns)

# Make sure 'review_text' and 'sentiment' exist
assert 'review_text' in df.columns and 'sentiment' in df.columns

# Map sentiment labels
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
reverse_label_map = {v: k for k, v in label_map.items()}

df['sentiment_label'] = df['sentiment'].map(label_map)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['review_text'], df['sentiment_label'], test_size=0.2, random_state=42
)

# ================== Prepare TF-IDF ==================
tfidf = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ================== Classifiers ==================
models = {
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LGBMClassifier": LGBMClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
}

# Voting Classifier (Ensemble)
voting_model = VotingClassifier(
    estimators=[
        ('xgb', models["XGBClassifier"]),
        ('lgbm', models["LGBMClassifier"]),
        ('lr', models["LogisticRegression"])
    ],
    voting='hard'  # or 'soft' if you want probability averaging
)

models["VotingClassifier"] = voting_model

results = {}

# ================== Train and Evaluate ==================
for name, model in models.items():
    print(f"\n====== Training {name} ======\n")
    
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_map.keys(), 
                yticklabels=label_map.keys())
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    results[name] = acc

# ================== Compare Final Results ==================
# Plot the comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = list(results.values())

sns.barplot(x=model_names, y=accuracies, palette='mako')
plt.title('Classifier Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
plt.show()
