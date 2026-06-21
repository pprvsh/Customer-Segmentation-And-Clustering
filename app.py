import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Customer Segmentation using K-Means Clustering")
st.markdown("Upload your dataset and segment customers into clusters.")

# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Data
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset must contain at least 2 numeric columns.")
        st.stop()

    st.sidebar.header("Clustering Settings")

    selected_features = st.sidebar.multiselect(
        "Select Features",
        numeric_cols,
        default=numeric_cols[:2]
    )

    k_clusters = st.sidebar.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    if len(selected_features) >= 2:

        X = df[selected_features]

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means
        kmeans = KMeans(
            n_clusters=k_clusters,
            random_state=42,
            n_init=10
        )

        clusters = kmeans.fit_predict(X_scaled)

        df["Cluster"] = clusters

        st.success(
            f"Successfully assigned customers into {k_clusters} distinct clusters!"
        )

        # Cluster Summary
        st.subheader("Cluster Distribution")
        st.write(df["Cluster"].value_counts().sort_index())

        # Clustered Data
        st.subheader("Clustered Dataset")
        st.dataframe(df.head())

        # Visualization
        st.subheader("Cluster Visualization")

        x_axis = st.selectbox(
            "Select X-axis",
            selected_features,
            index=0
        )

        y_axis = st.selectbox(
            "Select Y-axis",
            selected_features,
            index=1
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            df[x_axis],
            df[y_axis],
            c=df["Cluster"]
        )

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Customer Segments")

        plt.colorbar(scatter)

        st.pyplot(fig)

        # Download Option
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Clustered Data",
            data=csv,
            file_name="clustered_customers.csv",
            mime="text/csv"
        )

    else:
        st.warning("Please select at least two features.")

else:
    st.info("Upload a CSV file to begin clustering.")