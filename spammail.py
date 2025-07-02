import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# 1. Load data
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 2. Preprocess text
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)

# 3. Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['label_num']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Test accuracy
predictions = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, predictions))

# 7. Predict on new input
def predict_spam(msg):
    msg_clean = clean_text(msg)
    vector = vectorizer.transform([msg_clean])
    result = model.predict(vector)
    return "Spam 🚫" if result[0] == 1 else "Not Spam ✅"

# Try a sample
print(predict_spam("Congratulations! You've won a $1000 gift card. Click here!"))
print(predict_spam("Hey Parth, please submit the assignment by 5PM."))
