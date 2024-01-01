import text_utilities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

training_df, testing_df = text_utilities.go()

multinomial_vectorizer = CountVectorizer()
multinomial_classifier = MultinomialNB()

training_data_multinomial = multinomial_vectorizer.fit_transform(training_df['text'])
multinomial_classifier.fit(training_data_multinomial, training_df['author'])

multinomial_train_predictions = multinomial_classifier.predict(training_data_multinomial)
multinomial_test_predictions = multinomial_classifier.predict(multinomial_vectorizer.transform(testing_df['text']))

multinomial_train_accuracy = accuracy_score(training_df['author'], multinomial_train_predictions)
multinomial_test_accuracy = accuracy_score(testing_df['author'], multinomial_test_predictions)

bernoulli_vectorizer = CountVectorizer()
bernoulli_classifier = BernoulliNB()

training_data_bernoulli = bernoulli_vectorizer.fit_transform(training_df['text'])
bernoulli_classifier.fit(training_data_bernoulli, training_df['author'])

bernoulli_train_predictions = bernoulli_classifier.predict(training_data_bernoulli)
bernoulli_test_predictions = bernoulli_classifier.predict(bernoulli_vectorizer.transform(testing_df['text']))

bernoulli_train_accuracy = accuracy_score(training_df['author'], bernoulli_train_predictions)
bernoulli_test_accuracy = accuracy_score(testing_df['author'], bernoulli_test_predictions)

# Print the results
print("Multinomial Naive Bayes:")
print(f"Training Accuracy: {multinomial_train_accuracy}")
print(f"Testing Accuracy: {multinomial_test_accuracy}")

print("\nBernoulli Naive Bayes:")
print(f"Training Accuracy: {bernoulli_train_accuracy}")
print(f"Testing Accuracy: {bernoulli_test_accuracy}")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(training_df['author'].unique()),
                yticklabels=sorted(training_df['author'].unique()))
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(training_df['author'], multinomial_train_predictions, "Multinomial Naive Bayes - Training")
plot_confusion_matrix(training_df['author'], bernoulli_train_predictions, "Bernoulli Naive Bayes - Training")

plot_confusion_matrix(testing_df['author'], multinomial_test_predictions, "Multinomial Naive Bayes - Testing")
plot_confusion_matrix(testing_df['author'], bernoulli_test_predictions, "Bernoulli Naive Bayes - Testing")
