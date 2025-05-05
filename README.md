📊 Customer Loan Subscription Dashboard

Built with Streamlit | Machine Learning | Data Visualization | Clustering

This interactive dashboard predicts customer subscription to a bank's financial product (e.g., term deposit) using a Random Forest Classifier and segments customers into clusters using KMeans. Built with Streamlit and powered by `scikit-learn`, this app provides actionable insights for marketing and customer relationship teams.


🚀 Features

- ✅ Predict the probability of customer subscription based on key features.
- 📈 Visualize customer segments using dynamic charts (scatter, histogram, pie).
- 🧠 Machine Learning backend (Random Forest & KMeans Clustering).
- ⚙️ Interactive UI built with Streamlit sliders, forms, and dropdowns.
- 💡 Financial Health Score based on account balance.


## 📁 Project Structure
customer-subscription-dashboard ├── bank-full3.csv # Dataset used ├── app.py # Main Streamlit app ├── logo.png # (Optional) Logo for branding ├── requirements.txt # Dependencies └── README.md # Project documentation


🧠 ML Models

- Clustering: KMeans for customer segmentation.
- Classification: Random Forest to predict loan subscription likelihood.
- Scaler: StandardScaler for feature normalization.


🧪 Features Used

- Age  
- Balance  
- Duration (Last Contact Duration)  
- Campaign (Number of Contacts During Campaign)  
- Previous (Number of Previous Contacts)  
- Day of Contact  


💻 How to Run Locally

1. Clone the Repository
git clone https://github.com/your-username/loan-subscription-prediction.git
cd loan-subscription-prediction

2. Install requirements
   pip install -r requirements.txt

3. Run the app
   streamlit run streamlit_app.py

