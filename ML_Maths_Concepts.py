import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Define topics with detailed explanations and visualization functions
topics = {
    "Linear Algebra": {
        "description": """### **Linear Algebra: The Foundation of Data Representation**
Linear algebra is the backbone of machine learning, used for data storage and manipulation.

#### **Key Concepts:**
- **Vectors & Matrices:** Used to store multi-dimensional data.
- **Matrix Multiplication & Dot Product:** Computes relationships between features.
- **Eigenvalues & Eigenvectors:** Used in PCA (Principal Component Analysis).

#### **Applications:**
- **Neural Networks:** Stores weights and transforms input data.
- **Face Recognition:** PCA reduces image dimensions while retaining features.
- **Recommendation Systems:** Factorization techniques improve recommendations.
""",
        "plot": None
    },
    "Probability & Statistics": {
        "description": """### **Probability & Statistics: Understanding Uncertainty**
Probability and statistics help quantify randomness and variability in data.

#### **Key Concepts:**
- **Probability Distributions:** Models uncertainty (e.g., Normal distribution).
- **Bayes’ Theorem:** Predicts probabilities based on prior knowledge.
- **Expectation & Variance:** Measures data spread.

#### **Applications:**
- **Spam Filtering:** Determines whether an email is spam or not.
- **Stock Market Predictions:** Hidden Markov Models predict trends.
- **A/B Testing:** Helps businesses test features before rollout.
""",
        "plot": None
    },
    "Linear Regression & Optimization": {
        "description": """### **Linear Regression & Optimization: Predicting Trends**
Linear regression is a fundamental technique to find relationships between variables.

#### **Key Concepts:**
- **Ordinary Least Squares (OLS):** Minimizes the sum of squared errors.
- **Gradient Descent:** Iteratively adjusts weights to minimize error.

#### **Applications:**
- **House Price Prediction:** Estimates home values based on trends.
- **Demand Forecasting:** Retailers predict future sales.
""",
        "plot": "linear_regression"
    },
    "Probability Distributions": {
        "description": """### **Probability Distributions: Modeling Real-World Data**
Probability distributions describe how values are distributed.

#### **Key Concepts:**
- **Gaussian Distribution:** Used in anomaly detection.
- **Bernoulli & Binomial Distribution:** Helps in classification.
- **Poisson Distribution:** Models event occurrences over time.

#### **Applications:**
- **Fraud Detection:** Identifies unusual spending patterns.
- **Quality Control:** Detects defects in manufacturing.
""",
        "plot": "probability_distribution"
    },
    "Dimensionality Reduction (PCA)": {
        "description": """### **Dimensionality Reduction & Feature Engineering**
PCA helps reduce the number of features in high-dimensional data.

#### **Key Concepts:**
- **Principal Component Analysis (PCA):** Reduces high-dimensional data.
- **t-SNE:** Visualizes complex datasets in 2D/3D.

#### **Applications:**
- **Medical Diagnosis:** Reduces genetic data dimensions.
- **Image Compression:** Compresses images while retaining details.
""",
        "plot": "pca"
    },
    "Clustering & Unsupervised Learning": {
        "description": """### **Clustering & Unsupervised Learning**
Clustering groups data based on similarity without labels.

#### **Key Concepts:**
- **K-Means Clustering:** Groups similar data points.
- **Hierarchical Clustering:** Creates nested data clusters.

#### **Applications:**
- **Customer Segmentation:** Helps in personalized recommendations.
- **Anomaly Detection:** Used in cybersecurity and fraud detection.
""",
        "plot": "clustering"
    }
}

# Function to generate Linear Regression Plot
def plot_linear_regression():
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    Y = 2.5 * X + np.random.randn(50) * 3
    plt.figure(figsize=(6,4))
    plt.scatter(X, Y, label="Data Points")
    plt.plot(X, 2.5 * X, color="red", label="Best Fit Line")
    plt.xlabel("X (Input Feature)")
    plt.ylabel("Y (Target)")
    plt.title("Linear Regression: Best Fit Line")
    plt.legend()
    st.pyplot(plt)

# Function to generate Probability Distribution Plot
def plot_probability_distribution():
    np.random.seed(42)
    data = np.random.normal(loc=50, scale=15, size=1000)
    plt.figure(figsize=(6,4))
    sns.histplot(data, bins=30, kde=True)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Normal Distribution")
    st.pyplot(plt)

# Function to generate PCA Visualization
def plot_pca():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], title="PCA: Dimensionality Reduction")
    st.plotly_chart(df_pca)

# Function to generate Clustering Visualization
def plot_clustering():
    np.random.seed(42)
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.5)
    plt.figure(figsize=(6,4))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("K-Means Clustering Visualization")
    st.pyplot(plt)

# Streamlit UI
st.title("📊 Mathematics in ML")
st.write("Explore various mathematical concepts with explanations and interactive visualizations.")

# Dropdown for topic selection
selected_topic = st.selectbox("Choose a topic:", list(topics.keys()))

# Display topic description
st.markdown(topics[selected_topic]["description"])

# Display corresponding visualization if available
if topics[selected_topic]["plot"]:
    if topics[selected_topic]["plot"] == "linear_regression":
        plot_linear_regression()
    elif topics[selected_topic]["plot"] == "probability_distribution":
        plot_probability_distribution()
    elif topics[selected_topic]["plot"] == "pca":
        plot_pca()
    elif topics[selected_topic]["plot"] == "clustering":
        plot_clustering()
