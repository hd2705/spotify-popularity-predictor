# Hybrid Music Recommendation System
### 🎵 ML-Powered Song Discovery & Popularity Forecasting

This repository features a **Hybrid Recommendation Engine** developed for the **Advanced Machine Learning (CIS-550)** curriculum. The system integrates traditional content-based filtering with advanced regression modeling to solve the "popularity bias" and "cold start" problems common in music streaming services.

## 🚀 Executive Summary
Unlike standard recommenders that only look at sonic similarity, this system utilizes a dual-engine approach:
1.  **Popularity Prediction Engine:** Uses Gradient Boosting (XGBoost) and Random Forest to forecast a song's market success based on 21 distinct audio features.
2.  **Sonic Similarity Engine:** Employs K-Nearest Neighbors (KNN) and Cosine Similarity to find tracks with matching acoustic profiles.
3.  **Vibe-Based Filtering:** A custom interactive layer that maps qualitative user "vibes" (e.g., Happy, Hype, Chill) into quantitative feature constraints.

## 🛠️ Technical Tech Stack
* **Languages:** Python (Pandas, NumPy)
* **Modeling:** Scikit-learn, XGBoost, Random Forest
* **Evaluation Metrics:** MAE, MSE, R², ROC AUC
* **Visualization:** Matplotlib, Seaborn (Correlation Heatmaps, Feature Importance plots)

## 📊 Methodology & ML Pipeline

### 1. Data Engineering
* **Dataset:** Processed ~93,000 Spotify tracks.
* **Preprocessing:** Handled missing values, converted release dates to cyclical time features, and applied `MinMaxScaler` to normalize features like Loudness, Tempo, and Instrumentalness.
* **Filtering:** Applied a 70th percentile popularity threshold to define "high-impact" songs for model training.

### 2. The Hybrid Scoring Algorithm
The system uses a weighted scoring mechanism to rank recommendations:
$$Final Score = \alpha \cdot (ML Popularity Score) + (1 - \alpha) \cdot (Content Similarity Score)$$
*This allows the platform to balance "what sounds like your taste" with "what is currently trending."*

### 3. Model Performance & Feature Importance
Our analysis revealed that **Energy**, **Valence**, and **Danceability** are the primary predictors of a song's popularity. By comparing models, **XGBoost** provided the most robust performance with the lowest Mean Absolute Error (MAE).

## 📂 Repository Structure
* `notebooks/`: Includes `01_Data_Exploration_and_Cleaning.ipynb` and `Final.ipynb` (Model & Recommender).
* `data/`: Contains raw and pre-processed Spotify datasets.
* `reports/`: Includes the comprehensive **Group13_ML_ProjectReport.pdf** and the project presentation slides.

## 🎯 Business Outcomes & Impact
* **User Retention:** Personalized experiences reduce "choice fatigue," increasing platform engagement.
* **Data-Driven Curation:** Offers a framework for automated, high-quality playlist generation for streaming services.
* **Scalability:** The architecture is designed to be adaptable for podcasts, audiobooks, and other media recommendation tasks.

## 💻 Usage
1.  Clone the repository.
2.  Open `notebooks/Final.ipynb`.
3.  Run the cells to trigger the interactive "Vibe" prompt:
    * *Input Mood (Happy/Sad/Chill/Hype), Tempo (Slow/Fast), and Era (90s/2000s/etc.).*
    * *Receive a curated list of songs with both real and predicted popularity scores.*

---
**Author:** Hrushitha Darna  
**Academic Project:** Advanced Machine Learning 
