# Customer Segmentation Project

Segmenting 1,200 customers by **behaviour (RFM, discount use, channel) and demographics** with K-Means clustering in Python (scikit-learn), then turning each segment into a targeted marketing action.

> **Data note:** no dataset was provided with the task, so `generate_data.py` creates a realistic synthetic customer dataset (`data/customers.csv`). To use real data, keep the same columns and re-run `segmentation.py`.

## Project structure
| File | Purpose |
|------|---------|
| `generate_data.py` | Creates the synthetic dataset |
| `segmentation.py` | Cleaning check, scaling, choosing k, K-Means, profiling, charts |
| `data/customers.csv` | Input data (1,200 customers, 13 columns) |
| `outputs/` | Charts, `segment_summary.csv`, `customers_segmented.csv`, category preferences |

## How to run
```bash
pip install -r requirements.txt
python generate_data.py     # optional, data is already included
python segmentation.py
```

## Method
1. **Features (8):** Age, Annual Income, Recency, Frequency, Avg Order Value, Discount Usage %, Online Share %, Tenure.
2. **Scaling:** StandardScaler so no feature dominates.
3. **Choosing k:** elbow method + silhouette score for k = 2-8 (`outputs/01_choose_k.png`). k = 3 scored highest (0.355) but merges high-value and regular customers; **k = 4 (0.332)** is nearly as good and much more actionable, so it was chosen.
4. **Model:** K-Means (k = 4, `n_init=10`, `random_state=42`).
5. **Profiling:** segment averages, revenue share, category preferences, PCA 2-D view.

## Results
| Segment | Customers | Revenue share | Recency (days) | Orders/yr | Avg order | Discount use | Key traits |
|---------|-----------|---------------|----------------|-----------|-----------|--------------|------------|
| **Champions** | 17.8% | **58.9%** | 15 | 26 | $121 | 10% | Highest income, age ~38, loves Electronics |
| **Loyal Regulars** | 27.6% | 28.2% | 42 | 14 | $69 | 25% | Steady buyers, Home & Kitchen + Groceries |
| **Deal Seekers** | 25.7% | 7.6% | 70 | 8 | $35 | 58% | Youngest (~26), mostly online, Fashion + Beauty |
| **At-Risk / Lapsed** | 29.0% | 5.3% | 222 | 3 | $55 | 30% | Older, offline-leaning, haven't bought in ~7 months |

## Business insights and recommended actions
- **Champions (18% of customers) drive ~59% of revenue.** Protect them with early access, loyalty tiers and premium Electronics bundles. Never discount-blast them.
- **Loyal Regulars** are the best upgrade pool: cross-sell and subscription offers (Home & Kitchen, Groceries) to lift order value toward Champion levels.
- **Deal Seekers** buy only on discount. Use limited-time, app/online-only promos and bundles to raise basket size without hurting margin.
- **At-Risk customers (29%)** generate only ~5% of revenue. Run a low-cost win-back campaign (reminder + one targeted offer); drop non-responders from paid marketing.

## Charts
`01_choose_k` · `02_segments_pca` · `03_customer_vs_revenue_share` · `04_segment_profile_heatmap` · `05_category_preference` · `06_rfm_by_segment` (all in `outputs/`).

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Learning outcomes
Customer analytics, feature scaling, K-Means clustering, model selection (elbow/silhouette), segment profiling and targeted marketing insights.
