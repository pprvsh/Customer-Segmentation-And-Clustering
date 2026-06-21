# Customer Segmentation and Clustering

A data science project that segments customers into distinct groups based on their behavioral, demographic, and transactional attributes using unsupervised machine learning techniques. The goal is to help businesses understand customer profiles and design targeted marketing strategies.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

Customer segmentation is the practice of dividing a customer base into groups of individuals that share similar characteristics. This project applies clustering algorithms to identify meaningful customer segments, enabling:

- Targeted marketing campaigns
- Personalized product recommendations
- Improved customer retention strategies
- Better resource allocation

## Dataset

- Source: Kaggle 
- Size:`8951 

## Project Structure

customer-segmentation/
│
data/
  ├── raw/                # Original, immutable data
  └── processed/          # Cleaned and transformed data



├── notebooks/
 01_eda.ipynb              # Exploratory Data Analysis
 02_preprocessing.ipynb    # Cleaning & feature engineering
 03_clustering.ipynb       # Model building & evaluation
 04_visualization.ipynb    # Cluster profiling & plots



├── src/
data_preprocessing.py
clustering_model.py
visualization.py

├── reports/
figures/             # Generated charts and plots



├── requirements.txt
├── README.md
└── LICENSE


## Methodology

1. **Data Cleaning & Preprocessing**
   - Handling missing values and outliers
   - Encoding categorical variables
   - Feature scaling (StandardScaler / MinMaxScaler)

2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of key features
   - Correlation analysis
   - Visualizing customer attributes (age, income, spending patterns)

4. **Clustering Algorithms**
   - **K-Means** — partition-based clustering
   
5. **Determining Optimal Number of Clusters**
   - Elbow Method (inertia)

6. **Cluster Profiling**
   - Assigning descriptive labels to each segment (e.g., "High Income, Low Spenders")
   - Visualizing cluster characteristics

## Results


| Cluster | Profile | Size | Key Characteristics |
|---------|---------|------|----------------------|
| 0 | High-Value Customers | `<n>` | High income, high spending |
| 1 | Budget Shoppers | `<n>` | Low income, frequent purchases |
| 2 | Occasional Buyers | `<n>` | Moderate income, low frequency |
| 3 | New/Inactive Customers | `<n>` | Recently joined, low engagement |


## Technologies Used

- **Language:** Python 3.x
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn`, `plotly` — visualization
  - `scikit-learn` — clustering algorithms & preprocessing
  - `scipy` — hierarchical clustering / dendrograms
  - `jupyter` — interactive development

## Future Work

- Incorporate RFM (Recency, Frequency, Monetary) analysis
- Test additional algorithms (Gaussian Mixture Models, OPTICS)
- Build an interactive dashboard (Streamlit / Power BI / Tableau)
- Automate periodic re-segmentation with new data

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request, or open an issue to discuss proposed changes.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
