# PROJECT BY SAMUEL NNAMANI a.k.a Sammyst The Analyst.


# Import the necessary dependencies
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============== PAGE CONFIGURATION ==============
st.set_page_config(page_title="Customer Segmentation & Loan Prediction Dashboard", layout="wide")

# =============== LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("bank-full3.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# Load the data into a DataFrame
df = load_data()

# Show column names for debugging
# st.write("Available Columns:", df.columns.tolist())

# =============== PREPROCESSING ==============
# Convert target column 'y' to binary
df["y"] = df["y"].map({"yes":1, "no":0})

# Select relevant features for the model - mostly numeric features
features = ["age", "balance", "day", "duration", "campaign", "previous"]
X = df[features]
y = df["y"]

# Standardize the feature values
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns=features)

# Use the train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict the outcome
# rf_pred = rf_model.predict(X_test)
# Print the accuracy_score
# st.write("The model score of this algorithm is:", accuracy_score(rf_pred, y_test))

# ================== SIDEBAR CONTROLS ==================
st.sidebar.header("Dashboard Controls")

features_selected = st.sidebar.multiselect(
    "Features for Clustering:",
    options=features,
    default=['age', 'balance', 'day', 'duration', 'campaign', 'previous']
)

n_clusters = st.sidebar.slider("Number of Segments (Clusters):", min_value=2, max_value=10, value=4)

# ====================== CLUSTERING =======================
X_cluster = df[features_selected]
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

df['customer_segment'] = clusters

segment_selected = st.sidebar.multiselect(
    "Filter by Customer Segment:",
    options=sorted(df["customer_segment"].unique()),
    default=sorted(df["customer_segment"].unique())
)

df_selection = df[df["customer_segment"].isin(segment_selected)]


# ================= MAIN DASHBOARD ======================
st.title("Customer Segmentation & Subscription Prediction Dashboard")
st.subheader("Project by Samuel Nnamani a.k.a Sammyst The Analyst")
st.image("My_logo2.png", width=150)
# KPIs
st.subheader("Key Performance Indicators (KPIs)")
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(label="Total Customers", value=df_selection.shape[0])

with kpi2:
    st.metric(label="Average Balance", value=f"${df_selection['balance'].mean():.2f}")

with kpi3:
    st.metric(label="Average Age", value=f"{df_selection['age'].mean():.1f} years")

st.markdown("---")

# Visualizations
st.subheader("Visualizations")

# Histogram of Customer segments
st.markdown("### Customer Segments Distribution")
fig_segment = px.histogram(df_selection, x="customer_segment", color="customer_segment")
st.plotly_chart(fig_segment, use_container_width=True)

# Subscription rate
st.markdown("### Subscription Rate by Segment")
fig_subscribe = px.histogram(df_selection, x='customer_segment', color='y', barmode='group')
st.plotly_chart(fig_subscribe, use_container_width=True)

# Job distribution
st.markdown("### Job Distribution across Segments")
fig_job = px.histogram(df_selection, x='job', color="customer_segment", barmode='group')
st.plotly_chart(fig_job, use_container_width=True)

# Pie Charts
st.subheader("Pie Charts")

pie_col1, pie_col2 = st.columns(2)

with pie_col1:
    fig_pie_segment = px.pie(df_selection, names='customer_segment', title="Customer Segment Share")
    st.plotly_chart(fig_pie_segment, use_container_width=True)

with pie_col2:
    fig_pie_subscribe = px.pie(df_selection, names="y", title="Subscription (Yes/No) Share")
    st.plotly_chart(fig_pie_subscribe, use_container_width=True)

st.markdown("---")

# Scatter Plot
st.subheader("Scatter Plot of Segments")

scatter_x = st.selectbox("Select X-axis:", features_selected, index=0)
scatter_y = st.selectbox("Select Y-axis:", features_selected, index=1)

fig_scatter = px.scatter(
    df_selection,
    x=scatter_x,
    y=scatter_y,
    color="customer_segment",
    symbol="customer_segment",
    title=f"{scatter_x} vs {scatter_y} by Segment"
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ================== PREDICT CUSTOMER SUBSCRIPTION TO LOAN ====================
st.subheader("Predict Customer Subscription Probability")
st.write("This will predict the probability of a customer subscribing to a loan package by the financial institution")

with st.form('predict_form'):
    age_input = st.number_input("Age:", min_value=18, max_value=100, value=30)
    balance_input = st.number_input("Account Balance ($):", min_value=500, max_value=100000, value=500)
    day_input = st.number_input("Day of Month Contacted:", min_value=1, max_value=31, value=15)
    duration_input = st.number_input("Last Contact Duration (seconds):", min_value=1, max_value=5000, value=200)
    campaign_input = st.number_input("Number of Contacts During Campaign:", min_value=1, max_value=50, value=3)
    previous_input = st.number_input("Number of Previous Contacts:", min_value=0, max_value=50, value=0)

    submit = st.form_submit_button("Predict")

    if submit:
        try:
            # SANITIZE all inputs: remove whitespace, then converts all of them to numeric
            #age_input = int(str(age_input).strip())
            #balance_input = float(str(balance_input).strip())
            #day_input = int(str(day_input).strip())
            #duration_input = int(str(duration_input).strip())
            #campaign_input = int(str(campaign_input).strip())
            #previous_input = int(str(previous_input).strip())

            feature_names = ["age", "balance", "day", "duration", "campaign", "previous"]
            user_data = pd.DataFrame([[age_input, balance_input, day_input, duration_input, campaign_input, previous_input]],
                                     columns=feature_names)
            
            # Create user input DataFrame with all required features
            # user_data = pd.DataFrame({
            #     "age": [age_input],
            #     "balance": [balance_input],
            #     "day": [day_input],
            #     "duration": [duration_input],
            #     "campaign": [campaign_input],
            #     "previous": [previous_input]
            # })

            # Ensure the columns are in the same order as during training
            # user_data = user_data[rf_model.feature_names_in_]
            # Scale and ensure column alignment
            user_data_scaled_array = scaler.transform(user_data)
            user_data_scaled = pd.DataFrame(user_data_scaled_array, columns=feature_names)
            user_data_scaled = user_data_scaled[rf_model.feature_names_in_]
            
            # Predict
            pred_prob = rf_model.predict_proba(user_data_scaled)[0][1] * 100
            pred_class = "Subscribed" if pred_prob >= 50 else "Not Subscribed"

            st.metric(label="Predicted Subscription Probability", value=f"{pred_prob:.2f}%")
            st.success(f"Prediction: **{pred_class}**")

            # Financial Health Score
            st.subheader("💰 Financial Health Score")
            if balance_input >= 5000:
                score = "💎 Excellent"
            elif balance_input >= 2000:
                score = "🟢 Good"
            elif balance_input >= 0:
                score = "🟡 Average"
            else:
                score = "🔴 Poor"

            st.info(f"Financial Health: {score}")

        except Exception as e:
            st.error(f"🚨 Error in input: {e}. Please make sure all fields are numeric without strange characters.")
            st.write("Model expects these features:", list(rf_model.feature_names_in_))
