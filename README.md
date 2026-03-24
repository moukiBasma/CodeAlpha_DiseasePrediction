# Heart Disease Prediction

A machine learning project using Logistic Regression and Random Forest to predict heart disease from clinical data.

## Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 83.3% | 0.95 |
| Random Forest | **86.7%** | **0.95** |

Random Forest achieved the best performance with higher accuracy and balanced precision/recall, making it the preferred model for clinical prediction.

## Project Structure

```
CodeAlpha_DiseasePrediction/
├── data/heart.data          # UCI dataset (303 patients, 14 features)
├── notebook/heart_disease.ipynb
├── models/
│   ├── random_forest_model.pkl
│   └── logistic_model.pkl
└── outputs/                 # Confusion matrix, ROC, feature importance
```

## Tech Stack
- Models: Scikit-learn (Logistic Regression, Random Forest)
- Data: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Platform: Google Colab

## Quick Run
```python
# In Google Colab:
# 1. Upload notebook
# 2. Mount Drive & update paths
# 3. Run all cells → models train & save automatically
```

## Key Features
- Handles missing values (6 rows dropped from 303 total)
- Balanced class distribution (160 no disease, 137 disease) → no resampling required
- Top predictors identified: chest pain type (cp), thalassemia (thal), max heart rate (thalach), ST depression (oldpeak)

## Using the Trained Model
```python
import joblib

model = joblib.load('models/random_forest_model.pkl')
prediction = model.predict([[54,1,3,130,250,0,1,150,0,1.2,2,0,3]])

if prediction[0] == 1:
    print("Patient likely has heart disease")
else:
    print("Patient likely does not have heart disease")
```

## Future Work
- Implement neural networks for improved accuracy
- Deploy as a web application using Streamlit or Flask
- Add explainable AI components (SHAP/LIME) for model interpretability
- Expand to multi-class cardiovascular risk stratification

---
MIT License | CodeAlpha Internship Project
