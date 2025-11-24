#  AutoML Trainer & Predictor (Streamlit + scikit-learn)

This project is an end-to-end AutoML system built using Streamlit and scikit-learn.  
It allows users to upload any CSV dataset, automatically preprocess it, tune multiple machine learning models, select the best one, and export the final trained pipeline as a `.joblib` file.  
A separate predictor script loads this model and runs inference on new data.

---

##  Features

###  Automated ML Training
- Upload any CSV dataset  
- Select target column  
- Automatic task detection (classification/regression)  
- Preprocessing:
  - Numeric scaling + imputation  
  - Categorical encoding  
  - Text features using TF-IDF  
- Train/test/validation split  
- Hyperparameter tuning using `RandomizedSearchCV`  
- Automatic best-model selection  

###  Models Supported
- Logistic Regression  
- Random Forest  
- HistGradientBoosting  
- Linear Regression (for regression tasks)

---

##  Output
The app automatically generates:

- **`best_model_<timestamp>.joblib`**  
  Fully serialized pipeline for prediction  
- **`report_<timestamp>.json`**  
  Metrics, best params, and metadata  

---

##  Prediction Support
A separate Python script loads the trained model and predicts on new CSV input.  
Text + numeric preprocessing is automatically handled inside the pipeline.

---

##  Installation

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
