#  Adult Income Prediction using Artificial Neural Network (ANN)
Predict whether an individual's annual income exceeds **$50K** using an Artificial Neural Network (ANN).
# 📌 Project Overview
This project aims to predict whether a person's annual income is greater than **$50K** based on demographic and employment-related attributes.
An Artificial Neural Network (ANN) was developed using TensorFlow/Keras to perform binary classification on the Adult Income Dataset.
The project includes:
- Data preprocessing
- Feature engineering
- ANN model development
- Model evaluation
- Streamlit web application for real-time prediction
# 📂 Dataset

**Dataset:** Adult Income Dataset

**Source:** Kaggle – Adult Income Dataset

**Task:** Binary Classification

### Dataset Information

| Property | Value |
|----------|-------|
| Rows | 32,561 |
| Columns | 15 |
| Features | 14 |
| Target | income |
## Features
| Feature | Description |
|----------|-------------|
| age | Age of the individual |
| workclass | Employment type |
| fnlwgt | Census final weight |
| education | Highest education level |
| education.num | Numerical representation of education |
| marital.status | Marital status |
| occupation | Occupation |
| relationship | Family relationship |
| race | Race |
| sex | Gender |
| capital.gain | Capital gain |
| capital.loss | Capital loss |
| hours.per.week | Weekly working hours |
| native.country | Country of origin |
| income | Target variable |

---

# ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Loaded the dataset using Pandas.
- Checked dataset shape and data types.
- Encoded the target variable.
- Applied One-Hot Encoding to categorical features.
- Split the dataset into training and testing sets (80/20).
- Standardized numerical features using StandardScaler.

# 🧠 Model Architecture

The ANN model consists of:

- Input Layer
- Dense Layer (128 neurons, ReLU)
- Dropout Layer (0.3)
- Dense Layer (64 neurons, ReLU)
- Dropout Layer (0.2)
- Dense Layer (32 neurons, ReLU)
- Output Layer (1 neuron, Sigmoid)

### Model Configuration

| Parameter | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Batch Size | 32 |
| Epochs | 100|
| Early Stopping | Enabled |

---

# 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

# 📊 Model Performance

The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
Training history includes:
- Accuracy Curve
- Loss Curve
# 📁 Project Structure

Adult-Income-Prediction-ANN/
│
├── dataset/
│   └── adult.csv
├── model/
│   └── income_model.keras
├── images/
├── adultincome_prediction.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

# 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Explainable AI (SHAP)
- Deploy the model using Docker
- Cloud deployment

#  Author

**Sandy Reda**

AI & Data Science Student
