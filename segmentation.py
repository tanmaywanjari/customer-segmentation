"""Customer Segmentation with K-Means (RFM + demographics + behaviour)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
sns.set_theme(style="whitegrid")

df = pd.read_csv("data/customers.csv")
print("Rows:", len(df), "| Missing values:", int(df.isna().sum().sum()))

# 1. Feature selection & scaling
FEATURES = ["Age","Annual_Income_k","Recency_Days","Frequency","Avg_Order_Value",
            "Discount_Usage_Pct","Online_Share_Pct","Tenure_Months"]
X = StandardScaler().fit_transform(df[FEATURES])

# 2. Choose k (elbow + silhouette)
ks = range(2,9); inertia=[]; sil=[]
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(m.inertia_); sil.append(silhouette_score(X, m.labels_))
best_sil_k = list(ks)[int(np.argmax(sil))]
# k=3 has the top silhouette but merges "Champions" and "Loyal" customers into one group.
# k=4 is nearly as good statistically and far more useful for marketing, so we choose it.
best_k = 4
print("Silhouette by k:", {int(k): round(float(v),3) for k,v in zip(ks, sil)}, "| top silhouette k =", best_sil_k, "| chosen k =", best_k)
fig, ax = plt.subplots(1,2, figsize=(11,4))
ax[0].plot(ks, inertia, "o-"); ax[0].set(title="Elbow method", xlabel="k", ylabel="Inertia")
ax[1].plot(ks, sil, "o-", color="tab:green"); ax[1].set(title="Silhouette score", xlabel="k")
plt.tight_layout(); plt.savefig("outputs/01_choose_k.png", dpi=130); plt.close()

# 3. Final model
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["Cluster"] = km.labels_

# 4. Name segments from their profile
prof = df.groupby("Cluster")[FEATURES+["Total_Spend"]].mean()
names = {}
for c, r in prof.iterrows():
    if r.Recency_Days > 150: names[c] = "At-Risk / Lapsed"
    elif r.Total_Spend >= prof.Total_Spend.max()*0.95: names[c] = "Champions (High Value)"
    elif r.Discount_Usage_Pct > 45: names[c] = "Deal Seekers"
    else: names[c] = "Loyal Regulars"
df["Segment"] = df.Cluster.map(names)

# 5. Segment summary
summary = df.groupby("Segment").agg(Customers=("CustomerID","count"), Avg_Age=("Age","mean"),
    Avg_Income_k=("Annual_Income_k","mean"), Recency_Days=("Recency_Days","mean"),
    Frequency=("Frequency","mean"), Avg_Order_Value=("Avg_Order_Value","mean"),
    Discount_Pct=("Discount_Usage_Pct","mean"), Online_Pct=("Online_Share_Pct","mean"),
    Total_Revenue=("Total_Spend","sum")).round(1)
summary["Customer_Share_%"] = (summary.Customers/len(df)*100).round(1)
summary["Revenue_Share_%"] = (summary.Total_Revenue/summary.Total_Revenue.sum()*100).round(1)
summary = summary.sort_values("Total_Revenue", ascending=False)
summary.to_csv("outputs/segment_summary.csv"); print(summary.T.to_string())
df.to_csv("outputs/customers_segmented.csv", index=False)
top_cat = pd.crosstab(df.Segment, df.Preferred_Category, normalize="index").mul(100).round(1)
top_cat.to_csv("outputs/category_preference_by_segment.csv"); print(top_cat.to_string())

order = summary.index.tolist(); pal = sns.color_palette("Set2", len(order))
# 6. Visuals
pca = PCA(2, random_state=42); P = pca.fit_transform(X)
plt.figure(figsize=(7,5.5))
sns.scatterplot(x=P[:,0], y=P[:,1], hue=df.Segment, hue_order=order, palette=pal, s=22, alpha=.8)
plt.title(f"Customer segments (PCA, {pca.explained_variance_ratio_.sum():.0%} variance)")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout(); plt.savefig("outputs/02_segments_pca.png", dpi=130); plt.close()

fig, ax = plt.subplots(1,2, figsize=(11,4.5))
summary["Customer_Share_%"].plot.pie(ax=ax[0], autopct="%1.0f%%", colors=pal, ylabel="", title="Share of customers")
summary["Revenue_Share_%"].plot.pie(ax=ax[1], autopct="%1.0f%%", colors=pal, ylabel="", title="Share of revenue")
plt.tight_layout(); plt.savefig("outputs/03_customer_vs_revenue_share.png", dpi=130); plt.close()

z = (prof - prof.mean())/prof.std(); z.index = z.index.map(names)
plt.figure(figsize=(10,3.8))
sns.heatmap(z[FEATURES+["Total_Spend"]].loc[order], annot=prof.rename(index=names)[FEATURES+["Total_Spend"]].loc[order].round(0),
            fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={"label":"relative to average"})
plt.title("Segment profile (values = averages, colour = above/below overall)"); plt.ylabel("")
plt.tight_layout(); plt.savefig("outputs/04_segment_profile_heatmap.png", dpi=130); plt.close()

plt.figure(figsize=(9,4.5))
top_cat.loc[order].plot(kind="bar", stacked=True, colormap="tab20", ax=plt.gca())
plt.ylabel("% of segment"); plt.xlabel(""); plt.xticks(rotation=15); plt.title("Preferred category by segment")
plt.legend(bbox_to_anchor=(1.01,1), loc="upper left"); plt.tight_layout(); plt.savefig("outputs/05_category_preference.png", dpi=130); plt.close()

fig, ax = plt.subplots(1,3, figsize=(13,4))
for a,(col,t) in zip(ax,[("Recency_Days","Recency (days since last order)"),("Frequency","Orders per year"),("Total_Spend","Total spend")]):
    sns.boxplot(data=df, x="Segment", y=col, order=order, palette=pal, ax=a, hue="Segment", legend=False)
    a.set_title(t); a.set_xlabel(""); a.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.savefig("outputs/06_rfm_by_segment.png", dpi=130); plt.close()
print("Done. Charts saved in outputs/")
