# 🛍️ Customer Segmentation using K-Means Clustering

## 📌 Task 2 — Prodigy InfoTech ML Internship

## 📊 Dataset
Mall Customers Dataset — 200 customers with Annual Income and Spending Score.

## 🧠 Algorithm
K-Means Clustering with optimal K=5 (Elbow Method + Silhouette Score)

## 👥 Customer Segments Found
| Segment | Customers | Avg Income | Avg Spending |
|---|---|---|---|
| 💎 Premium Loyalists | 39 | $86.5k | 82.1 |
| 💼 High Earners, Low Spend | 35 | $88.2k | 17.1 |
| 🛍️ Budget Spenders | 22 | $25.7k | 79.4 |
| 💤 Low Engagement | 23 | $26.3k | 20.9 |
| ⚖️ Average Customers | 81 | $55.3k | 49.5 |

## 🛠️ Libraries Used
- pandas, numpy, scikit-learn, matplotlib, seaborn

## ▶️ How to Run
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
python kmeans_customer_segmentation.py