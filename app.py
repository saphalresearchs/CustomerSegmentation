import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

# Load trained KMeans model
model = joblib.load("segmentation.pkl")

df= pd.read_csv("Mall_Customers.csv")
df.drop(columns=["CustomerID","Gender","Age"], axis=1)

# Streamlit UI
st.title("💰 Customer Segmentation using K-Means")
st.write("Predict customer segments based on **Annual Salary** and **Spending Score**.")

df_clustered = df[["Annual Income (k$)", "Spending Score (1-100)"]]
df_clustered["Cluster"] = model.predict(df_clustered)


# Define segment labels based on clustering patterns
segment_labels = {
    0: "Moderate Income, Balanced Spend",
    1: "High Income, High Spend",
    2: "Low Income, High Spend",
    3: "High Income, Low Spend",
    4: "Low Income, Low Spend"
}

# Map clusters to labels
df_clustered["Segment"] = df_clustered["Cluster"].map(segment_labels)


# User input
annual_salary = st.number_input("Enter Annual Salary (in $1000)", min_value=0, step=1, value=50)
spending_score = st.number_input("Enter Spending Score (0-100)", min_value=0, max_value=100, step=1, value=50)

# Predict customer segment
if st.button("Predict Segment"):
    input_data = np.array([[annual_salary, spending_score]])
    cluster = model.predict(input_data)[0]
    segment_name = segment_labels.get(cluster, "Unknown Segment ❓")
    st.success(f"🛍️ The customer belongs to **{segment_name}** (Cluster {cluster})")


# Add new customer data to a copy of the DataFrame for visualization
# df_viz = df.copy()
# df_viz = pd.concat([df_viz, pd.DataFrame({'Annual Income (k$)': [annual_salary],
#                                           'Spending Score (1-100)': [spending_score],
#                                           'Cluster': [cluster],
#                                           'Segment': [segment_name]})])




# Plotly interactive visualization
st.subheader("📈 Interactive Segmentation Chart")
fig = px.scatter(df_clustered, x="Annual Income (k$)", y="Spending Score (1-100)", 
                 color="Segment",
                 title="Customer Segments",
                 hover_data=["Annual Income (k$)", "Spending Score (1-100)", "Segment"])
st.plotly_chart(fig, use_container_width=True)


fig.add_scatter(x=[annual_salary], y=[spending_score],
                mode="markers", marker=dict(size=15, color="black", symbol="star"),
                name="New Customer")
st.plotly_chart(fig, use_container_width=True)
