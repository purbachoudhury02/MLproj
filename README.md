# Student Exam Performance Predictor

An end-to-end machine learning pipeline that predicts a student's math score based on their other exam scores and background factors.Student performance is influenced by both academic and demographic factors. This project predicts **math_score** using a student's reading score, writing score, and background attributes (gender, ethnicity, parental education level, lunch type, and test preparation course completion).

## Dataset

- Source: Student performance dataset (`https://www.kaggle.com/datasets/spscientist/students-performance-in-exams`)
- Target variable: `math_score`
- Features used:
  - Numerical: `reading_score`, `writing_score`
  - Categorical: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`

## Project Structure

| File/Folder | Description |
|---|---|
| `src/components/data_ingestion.py` | Reads raw data, performs train/test split (80/20), saves to `artifacts/` |
| `src/components/data_transformation.py` | Builds preprocessing pipeline — median imputation + scaling for numerical features, mode imputation + one-hot encoding + scaling for categorical features, combined via `ColumnTransformer` |
| `src/components/model_trainer.py` | Trains and benchmarks multiple regression models with hyperparameter tuning, selects the best performer by test R² score |
| `src/pipeline/predict_pipeline.py` | Loads the saved model and preprocessor to generate predictions on new input |
| `app.py` / `application.py` | Flask web application for interactive predictions |
| `notebook/` | EDA and experimentation notebooks |
| `artifacts/` | Saved train/test splits, preprocessor object, and trained model (generated at runtime) |

## Approach

Rather than training a single model, seven regression algorithms were benchmarked with hyperparameter tuning via grid search:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- XGBoost Regressor
- CatBoost Regressor

The best-performing model (by test R² score) is automatically selected and saved for deployment.

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost
- **Web Framework:** Flask
- **Deployment:** Local Flask app (form-based prediction interface)

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000/` and navigate to the prediction page.


