import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the data
df = pd.read_csv('data.csv')

# Preprocess and split the data
X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train_counts, y_train)

# Evaluate the model
predictions = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, predictions)

# Streamlit app
st.title('Financial Market Sentiment Analysis')

st.write(f"Model Accuracy: {accuracy*100:.2f}%")

user_input = st.text_area("Enter a sentence about the financial market", "The economic indicators show growth")

# Predict sentiment
if st.button('Analyze Sentiment'):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    st.write(f"Predicted sentiment: {prediction[0]}")
