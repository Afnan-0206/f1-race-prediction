# 🏎️ Formula 1 Race Position Prediction

This project focuses on predicting Formula 1 race finishing positions using machine learning and 70+ years of historical race data.

---

## 📌 Overview

Formula 1 is a highly data-driven sport where race outcomes depend on multiple factors such as grid position, qualifying performance, driver form, and team strength.

In this project, we built a predictive model to estimate driver finishing positions for future races.

---

## 🧠 Approach

- Used **LightGBM (Gradient Boosting)** for regression
- Implemented **time-based cross-validation** to avoid data leakage
- Focused on **feature engineering** to capture race dynamics

---

## 🔧 Key Features

- Grid-based features (starting position importance)
- Qualifying performance metrics
- Driver recent form (rolling averages)
- Race-relative ranking features
- Championship context (driver & constructor points)

---

## 📊 Model Performance

- Evaluation Metric: **Mean Absolute Error (MAE)**
- Achieved MAE: ~**3.2 – 3.5**
- Outperformed baseline model significantly

---

## 📁 Tech Stack

- Python
- Pandas, NumPy
- LightGBM
- Scikit-learn

---

## 🚀 Future Improvements

- Ensemble models (LightGBM + XGBoost)
- Advanced feature engineering
- Hyperparameter tuning
- Real-time prediction dashboard

---

## 👨‍💻 Contributors

- Afnan  
- Shrinidhi  

---

## ⭐ Conclusion

This project demonstrates how machine learning can be applied to complex real-world sports data to generate meaningful predictions and insights.
