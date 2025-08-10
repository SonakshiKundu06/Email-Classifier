# Objective: Classify emails as "spam" or "ham" based on text features.
# •	Dataset: SpamAssassin public dataset or UCI SMS Spam Collection
# •	Model: Naive Bayes / Logistic Regression
# •	Preprocessing: Tokenization, TF-IDF Vectorizer
# •	Visualization:
# o	Word cloud for spam vs ham
# o	Confusion matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

# 1. Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# 2. Encode Labels
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Text Preprocessing & TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

nb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# 6. Predict & Evaluate
y_pred_nb = nb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Print Classification Reports
print("Naive Bayes Report:\n", classification_report(y_test, y_pred_nb))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_lr))

# 7. Confusion Matrix
def plot_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_conf_matrix(y_test, y_pred_nb, "Naive Bayes")
plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression")

# 8. Word Clouds
spam_words = ' '.join(df[df['label']=='spam']['message'])
ham_words = ' '.join(df[df['label']=='ham']['message'])

spam_wc = WordCloud(width=600, height=400, background_color='white').generate(spam_words)
ham_wc = WordCloud(width=600, height=400, background_color='white').generate(ham_words)

plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Spam Word Cloud")
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title("Ham Word Cloud")
plt.show()