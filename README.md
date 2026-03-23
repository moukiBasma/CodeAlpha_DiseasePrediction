# ❤️ Heart Disease Prediction

A machine learning project using **Logistic Regression** and **Random Forest** to predict heart disease from clinical data.

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Logistic Regression** | 83.3% | 0.95 |
| **Random Forest** ⭐ | **86.7%** | **0.95** |

> ✅ **Best Model:** Random Forest — higher accuracy with balanced precision/recall.

## 📁 Project Structure

```
CodeAlpha_DiseasePrediction/
├── data/heart.data          # UCI dataset (303 patients, 14 features)
├── notebook/heart_disease.ipynb
├── models/
│   ├── random_forest_model.pkl
│   └── logistic_model.pkl
└── outputs/                 # Confusion matrix, ROC, feature importance
```

## 🛠️ Tech Stack
- **Models:** Scikit-learn (Logistic Regression, Random Forest)
- **Data:** Pandas, NumPy
- **Visuals:** Matplotlib, Seaborn
- **Platform:** Google Colab

## 🚀 Quick Run
```python
# In Google Colab:
1. Upload notebook
2. Mount Drive & update paths
3. Run all cells → models train & save automatically
```

## 🔑 Key Features
- Handles missing values (6 rows dropped)
- Balanced classes → no resampling needed
- Top predictors: `cp` (chest pain), `thal`, `thalach`, `oldpeak`

## 💾 Use Trained Model
```python
import joblib
model = joblib.load('models/random_forest_model.pkl')
prediction = model.predict([[54,1,3,130,250,0,1,150,0,1.2,2,0,3]])
print("⚠️ Disease" if prediction[0]==1 else "✅ Healthy")
```

## 🔮 Next Steps
- [ ] Add neural networks
- [ ] Deploy as web app (Streamlit)
- [ ] Include explainable AI (SHAP)

---
*MIT License | CodeAlpha Internship Project* 
