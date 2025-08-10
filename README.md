📧 Email Classifier — Spam vs Ham
🎯 Objective
Classify messages as spam or ham (not spam) based on text features, using Naive Bayes and Logistic Regression models.

📂 Dataset
Source: UCI SMS Spam Collection

Direct Link Used: sms.tsv

Contains 5,574 messages labeled as ham or spam.

🧠 Models Implemented
Multinomial Naive Bayes

Logistic Regression (max_iter=1000)

🔄 Preprocessing Steps
Label encoding (ham → 0, spam → 1)

TF-IDF Vectorization with English stopwords removed

Train-Test split (80% train, 20% test)

📊 Visualizations Generated
Word Clouds

Spam messages word frequency

Ham messages word frequency

Confusion Matrices

Naive Bayes

Logistic Regression

📈 Evaluation Metrics
The script prints a classification report for each model, including:

Accuracy

Precision

Recall

F1-score

🖼 Output Samples

<img width="994" height="683" alt="Screenshot 2025-08-11 002123" src="https://github.com/user-attachments/assets/f2f4c42a-58fd-46ec-8a56-75e2dafff644" />
<img width="978" height="670" alt="Screenshot 2025-08-11 002143" src="https://github.com/user-attachments/assets/d28a4901-ffd3-45cf-822f-5feb1f33eedb" />

