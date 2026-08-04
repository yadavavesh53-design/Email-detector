import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("emails.csv")

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test with user input
email = input("Enter an email message: ")

email_vector = vectorizer.transform([email])
prediction = model.predict(email_vector)

print("Prediction:", prediction[0])
import joblib

model = MultinomialNB()
model.fit(X_train, y_train)

joblib.dump(model, "models/spam_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model saved successfully!")