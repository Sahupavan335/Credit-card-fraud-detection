# 💳 Credit Card Fraud Detection

## 📝 1. Project Title
Credit Card Fraud Detection using Machine Learning

## 📌 2. Brief One-Line Summary
A machine learning-based system to detect fraudulent credit card transactions using anonymized financial data.

## 📖 3. Overview
Credit card fraud is a major financial crime, causing significant losses to banks and customers. This project applies supervised machine learning algorithms to detect fraudulent transactions based on anonymized transaction attributes. The goal is to build a robust classification system that can handle imbalanced datasets and provide accurate fraud detection.

## ❓ 4. Problem Statement
The dataset is highly imbalanced, with fraudulent transactions representing less than 1% of the total. The challenge is to identify frauds effectively while minimizing false positives.

## 📊 5. Dataset
- **Source:** Kaggle (Credit Card Fraud Detection Dataset 2023)  
- **Records:** ~550,000 transactions by European cardholders  
- **Features:**  
  - `V1–V28`: Anonymized transaction features (PCA-transformed)  
  - `Amount`: Transaction amount  
  - `Class`: Target label (0 = Legitimate, 1 = Fraudulent)

## 🛠️ 6. Tools and Technologies
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, lightgbm, streamlit, pickle  
- **Environment:** Jupyter Notebook, Streamlit web app  
- **Version Control:** Git, GitHub  

## ⚙️ 7. Methods
1. Data preprocessing (scaling, handling imbalance)  
2. Model training using:  
   - RandomForestClassifier  
   - LightGBMClassifier  
3. Model evaluation using precision, recall, F1-score, ROC-AUC  
4. Deployment of prediction app with Streamlit  

## 🔑 8. Key Insights
- Fraudulent transactions are extremely rare (~0.1–0.2%).  
- LightGBM performed slightly better in handling class imbalance compared to RandomForest.  
- Transaction amount and certain V-features strongly correlate with fraud cases.  

## 📊 9. Dashboard / Model / Output
- **Streamlit App**: Interactive web application for predicting fraud.  
- **Features**:  
  - Single transaction input  
  - CSV file upload (supports large files up to ~400MB)  
  - Downloadable results with predictions & probabilities  

## 🚀 10. How to Run this Project?
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
2. Install dependencies
    ```bash
    pip install -r requirements.txt
3. Run the code
   ```bash
   streamlit run app.py for streamlit app
   python flask_app.py for flask app

## 11. Result & Conclusion

The models achieved high ROC-AUC (>0.95) despite class imbalance.

LightGBM gave better recall (important for fraud detection) while RandomForest provided balanced performance.

A deployed Streamlit app allows real-time and batch predictions.

## 12. Future Work

- Integrate deep learning models (e.g., autoencoders for anomaly detection).

- Deploy on cloud platforms (AWS/GCP/Heroku).

- Add merchant category & geolocation analysis (if available).

- Implement model monitoring for concept drift in fraud patterns.

## 13. Author & Contact

- Author: Sahu Pavan
- Email: sahupavan335@gmail.com

- LinkedIn: linkedin.com/in/sahu-pavan-a01633266

- GitHub: 

