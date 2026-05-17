"""
================================================================
  K-Means Customer Segmentation — Complete Beginner Guide
  Dataset: Mall_Customers.csv
  Author: ved patel | Task:-2
================================================================
"""

# ================================================================
# STEP 1: INSTALL LIBRARIES 
# ================================================================
# pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

# ================================================================
# STEP 2: IMPORT LIBRARIES
# ================================================================

import pandas as pd              # Data loading and manipulation
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt  # Plotting graphs
import seaborn as sns            # Beautiful statistical plots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler   # Feature scaling
from sklearn.cluster import KMeans                 # K-Means algorithm
from sklearn.metrics import silhouette_score       # Evaluate quality


# ================================================================
# STEP 3: LOAD DATASET
# ================================================================
# If using CSV:
df = pd.read_csv("Mall_Customers.csv")

# If using Excel (.xlsx):
# df = pd.read_excel("Mall_Customers.xlsx")

print("=" * 55)
print("  MALL CUSTOMER SEGMENTATION — K-MEANS CLUSTERING")
print("=" * 55)
print(f"\n✅ Dataset Loaded! Shape: {df.shape}")
print(f"   → {df.shape[0]} customers, {df.shape[1]} columns")

print("\n📋 First 5 rows:")
print(df.head().to_string())

print("\n📊 Column Names:", df.columns.tolist())


# ================================================================
# STEP 4: DATA PREPROCESSING
# ================================================================

# 4a. Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())
# Output should be 0 for all — no cleaning needed here!

# 4b. Basic statistics
print("\n📈 Dataset Statistics:")
print(df.describe().round(2).to_string())

# 4c. Select features for clustering
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features].copy()

print(f"\n✅ Selected Features: {features}")

# 4d. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features scaled using StandardScaler")
print("   Before scaling — Income range:", X["Annual Income (k$)"].min(), "to", X["Annual Income (k$)"].max())
print("   After scaling  — Income range:", round(X_scaled[:,0].min(), 2), "to", round(X_scaled[:,0].max(), 2))


# ================================================================
# STEP 5: FIND OPTIMAL K — ELBOW METHOD + SILHOUETTE SCORE
# ================================================================

print("\n⏳ Computing Elbow Method and Silhouette Scores...")
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    score = silhouette_score(X_scaled, km.labels_)
    silhouette_scores.append(score)
    print(f"   K={k:2d} → Inertia: {km.inertia_:7.2f}  |  Silhouette: {score:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Finding Optimal K — Elbow & Silhouette", fontsize=14, fontweight="bold")

# Elbow
axes[0].plot(list(K_range), inertia, "bo-", linewidth=2.5, markersize=8)
axes[0].axvline(x=5, color="red", linestyle="--", linewidth=2, label="K=5 (Optimal)")
axes[0].fill_between(list(K_range), inertia, alpha=0.1, color="blue")
axes[0].set_title("Elbow Method", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Silhouette
bar_colors = ["#FF4444" if k == 5 else "#4ECDC4" for k in K_range]
axes[1].bar(list(K_range), silhouette_scores, color=bar_colors, edgecolor="black", linewidth=0.5)
for i, (k, s) in enumerate(zip(K_range, silhouette_scores)):
    axes[1].text(k, s + 0.005, f"{s:.3f}", ha="center", va="bottom", fontsize=8)
axes[1].set_title("Silhouette Scores (Higher = Better)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Elbow chart saved as 'elbow_silhouette.png'")


# ================================================================
# STEP 6: APPLY K-MEANS WITH OPTIMAL K = 5
# ================================================================

print("\n🚀 Training K-Means with K=5...")
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"✅ K-Means trained! Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")

# Decode cluster centers back to original scale
centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
print("\n📍 Cluster Centers (original scale):")
for i, c in enumerate(centers_orig):
    print(f"   Cluster {i}: Income = {c[0]:.1f}k$, Spending Score = {c[1]:.1f}")

# Assign human-readable names based on centers
cluster_names = {}
for i, c in enumerate(centers_orig):
    income, spend = c[0], c[1]
    if income > 70 and spend > 70:
        cluster_names[i] = "💎 Premium Loyalists"
    elif income > 70 and spend < 40:
        cluster_names[i] = "💼 High Earners, Low Spend"
    elif income < 45 and spend > 60:
        cluster_names[i] = "🛍️ Budget Spenders"
    elif income < 45 and spend < 40:
        cluster_names[i] = "💤 Low Engagement"
    else:
        cluster_names[i] = "⚖️ Average Customers"

df["Customer_Segment"] = df["Cluster"].map(cluster_names)

print("\n📦 Customer Distribution:")
print(df["Customer_Segment"].value_counts())


# ================================================================
# STEP 7: VISUALIZATIONS
# ================================================================

# ── Visualization 1: Main Cluster Scatter Plot ───────────────────
colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA", "#34D399"]
segments = list(df["Customer_Segment"].unique())
seg_color = {seg: colors[i] for i, seg in enumerate(segments)}

plt.figure(figsize=(12, 8))
for seg, grp in df.groupby("Customer_Segment"):
    plt.scatter(grp["Annual Income (k$)"], grp["Spending Score (1-100)"],
                c=seg_color[seg], label=seg, alpha=0.85, s=80,
                edgecolors="white", linewidths=0.5)

# Plot centroids
plt.scatter(centers_orig[:, 0], centers_orig[:, 1],
            c="white", s=300, marker="*", edgecolors="black",
            linewidths=1.0, zorder=10, label="⭐ Centroids")

plt.title("Customer Segments — Annual Income vs Spending Score",
          fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Annual Income (k$)", fontsize=12)
plt.ylabel("Spending Score (1-100)", fontsize=12)
plt.legend(fontsize=10, loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Scatter plot saved as 'cluster_scatter.png'")


# ── Visualization 2: Cluster Summary Bar Charts ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cluster Profiles — Average Values by Segment", fontsize=14, fontweight="bold")

summary = df.groupby("Customer_Segment")[["Annual Income (k$)", "Spending Score (1-100)", "Age"]].mean().round(1)
bar_cols = [seg_color[s] for s in summary.index]

for ax, col in zip(axes, ["Annual Income (k$)", "Spending Score (1-100)", "Age"]):
    bars = ax.bar(range(len(summary)), summary[col], color=bar_cols, edgecolor="black", lw=0.5)
    ax.set_title(f"Avg {col}", fontweight="bold")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([s.split()[-1] for s in summary.index], rotation=30, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, summary[col]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("cluster_profiles.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Profile chart saved as 'cluster_profiles.png'")


# ================================================================
# STEP 8: INTERPRETATION — WHAT EACH CLUSTER MEANS
# ================================================================
print("\n" + "=" * 60)
print("  📋 CLUSTER INTERPRETATION (Business Insights)")
print("=" * 60)

interpretations = {
    "💎 Premium Loyalists":
        "HIGH income + HIGH spending. VIP customers! Reward them\n"
        "   with loyalty programs, early access, and premium offers.",
    "💼 High Earners, Low Spend":
        "HIGH income + LOW spending. Big potential! They can afford\n"
        "   more but aren't buying. Target with luxury/exclusive deals.",
    "🛍️ Budget Spenders":
        "LOW income + HIGH spending. Enthusiastic but budget-limited.\n"
        "   Target with discounts, EMI offers, and flash sales.",
    "💤 Low Engagement":
        "LOW income + LOW spending. Hardest to convert.\n"
        "   Try value packs, combo deals, or loyalty entry programs.",
    "⚖️ Average Customers":
        "MODERATE income + MODERATE spending. The backbone!\n"
        "   Keep them engaged with regular promotions and membership benefits."
}
for seg, desc in interpretations.items():
    count = df[df["Customer_Segment"]==seg].shape[0] if seg in df["Customer_Segment"].values else 0
    print(f"\n  {seg} ({count} customers)")
    print(f"   → {desc}")


# ================================================================
# STEP 9: SAVE RESULTS
# ================================================================
output_cols = ["CustomerID", "Gender", "Age", "Annual Income (k$)",
               "Spending Score (1-100)", "Cluster", "Customer_Segment"]

# Save as CSV
df[output_cols].to_csv("customer_segments_output.csv", index=False)
print("\n✅ Results saved to 'customer_segments_output.csv'")

# Save as Excel
df[output_cols].to_excel("customer_segments_output.xlsx", index=False)
print("✅ Results saved to 'customer_segments_output.xlsx'")

# Final summary table
print("\n📊 FINAL CLUSTER SUMMARY:")
final_summary = df.groupby("Customer_Segment").agg(
    Count=("CustomerID", "count"),
    Avg_Income=("Annual Income (k$)", "mean"),
    Avg_Spending=("Spending Score (1-100)", "mean"),
    Avg_Age=("Age", "mean")
).round(1)
print(final_summary.to_string())

print("\n🎉 Pipeline complete! All files saved.")
print(f"   🎯 Final Silhouette Score: {silhouette_score(X_scaled, df['Cluster']):.4f}")
print("   (Score closer to 1.0 = well-separated clusters)")
