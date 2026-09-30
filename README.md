# Spam Email Classifier

Trains a model to detect spam messages from their text content, comparing two
classic techniques: **Multinomial Naive Bayes** and a **Linear Support Vector
Machine (SVM)**.

## Project Structure

```
spam_classifier/
├── data/spam.csv            # label (ham/spam), text
├── spam_classifier.py       # cleaning, TF-IDF pipeline, training, evaluation
├── train.py                 # trains both models, prints metrics, saves best
├── predict.py               # classify your own messages
├── requirements.txt
├── models/                  # saved model (created by train.py)
└── outputs/                 # confusion matrix plot (created by train.py)
```

## How It Works

1. **Cleaning** - lowercase, replace URLs and numbers with tokens (`urltoken`,
   `numtoken`, since spam is full of links and phone numbers), strip punctuation.
2. **Features** - TF-IDF on unigrams and bigrams, English stop words removed.
3. **Models** - Naive Bayes and Linear SVM trained on the same stratified
   80/20 train/test split.
4. **Evaluation** - accuracy, precision, recall, F1 and a confusion matrix.
   Because only ~13% of messages are spam, accuracy alone is misleading, so the
   best model is chosen by **F1 on the spam class**.

## Usage

```bash
pip install -r requirements.txt
python train.py                      # train + compare + save best model
python predict.py                    # built-in examples, then interactive mode
python predict.py "You won a free prize, click here"
```

## Results (20% held-out test set, 1,034 messages)

| Model        | Accuracy | Precision | Recall | F1    |
|--------------|----------|-----------|--------|-------|
| Naive Bayes  | 0.9874   | 0.9758    | 0.9237 | 0.949 |
| Linear SVM   | 0.9845   | 0.9528    | 0.9237 | 0.938 |

Naive Bayes is saved as the final model. Both models miss about 8% of spam
(recall) while rarely flagging genuine messages, which is the right trade-off
for a spam filter, since a false positive (losing a real message) is worse than
a missed spam.

## Dataset

Uses the public **SMS Spam Collection** (5,572 labelled messages, 5,169 after
removing duplicates). It is short-text messages rather than full emails, but the
technique is identical. To use email data (e.g. the Enron or SpamAssassin
corpora), put it in `data/spam.csv` with columns `label` (`ham`/`spam`) and
`text`, then re-run `python train.py`.

## Possible Extensions

- Tune hyperparameters with `GridSearchCV` and use cross-validation.
- Add engineered features (message length, count of `!`, uppercase ratio).
- Try Logistic Regression or a transformer-based model for comparison.
- Wrap it in a Flask or Streamlit app for a live spam checker.
