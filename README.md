# NLP with Disaster Tweets

Machine learning project for binary text classification on the **Real or Not? NLP with Disaster Tweets** dataset.

The project compares multiple models after standard NLP preprocessing, then generates Kaggle-style submission files.

## Project Goals

- Clean and normalize tweet text.
- Build a reproducible train/validation pipeline.
- Compare several baseline and non-linear classifiers.
- Select a final model based on validation performance and generalization.

## Dataset

- Train data: `train.csv`
- Test data: `test.csv`
- Submission template: `sample_submission.csv`

Original competition: [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

## Main Notebook

- `nlp_disaster_tweets.ipynb`

The notebook includes:

1. Data loading
2. Text preprocessing (regex cleaning, lowercasing, tokenization, stopword removal)
3. Feature extraction with `CountVectorizer`
4. Model training and evaluation
5. Test prediction and submission file generation

## Key Findings (Applied in Code Comments)

- The target classes are close to balanced, so no over/undersampling was required.
- **Logistic Regression with `C=0.1`** gave the best validation behavior among tested classical models.
- Larger LR `C` values increased training accuracy but reduced validation performance (overfitting signal).
- RBF-SVM was competitive but slightly below the best LR setup on validation.
- Decision Trees showed clear overfitting (very high train score vs lower validation score).

## Model Outputs

The notebook can generate:

- `submission_lr.csv`
- `submission_lr_op.csv` (recommended model, LR with `C=0.1`)
- `submission_svm.csv`
- `submission_svm_RBF.csv`

Existing generated submission files are stored in `outputs/`.

## Project Files

- `project_report.pdf`: final project report (recommended for GitHub viewers)
- `project_report.doc`: editable report source
- `source_data/socialmedia-disaster-tweets-DFE.csv`: original uncleaned source file

## Environment

Suggested Python version: `3.9+`

Install dependencies:

```bash
pip install pandas numpy nltk scikit-learn matplotlib tensorflow keras
```

NLTK resources used:

- `stopwords`
- `punkt`

## How To Run

1. Open the notebook.
2. Run cells from top to bottom.
3. Confirm generated CSV submission files in the project root or move them to `outputs/`.

## Notes for GitHub Upload

- Keep the notebook as the main experiment log.
- Keep this `README.md` as the project entry point.
- Keep generated result CSV files in `outputs/` for clarity.
- Optional cleanup before pushing:
  - clear heavy notebook outputs for a lighter repository
