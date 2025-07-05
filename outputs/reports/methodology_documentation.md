
# E-commerce Market Research & Customer Segmentation Analysis
## Methodology Documentation

**Author:** Bryant M.  
**Date:** July 2025  
**Project:** Portfolio Project 2 - Online Retail Analysis

---

## Overview

This document outlines the comprehensive methodology used for analyzing e-commerce retail data, including customer segmentation, market trends analysis, and revenue optimization strategies.

## 1. Data Processing Pipeline

### 1.1 Data Source
- **Dataset:** UCI Online Retail Dataset
- **Format:** Excel (.xlsx)
- **Size:** ~540,000 transactions
- **Period:** December 2010 - December 2011
- **Scope:** UK-based non-store online retail

### 1.2 Data Quality Assessment
```python
# Key quality checks performed
- Missing value analysis
- Duplicate transaction detection
- Outlier identification (price, quantity)
- Date range validation
- Customer ID completeness
- Product code validity
```

### 1.3 Data Cleaning Process
1. **Missing Data Handling:**
   - Removed transactions without CustomerID (B2B transactions)
   - Filled missing product descriptions with "UNKNOWN PRODUCT"

2. **Outlier Treatment:**
   - Removed negative quantities (returns/cancellations)
   - Removed zero or negative unit prices
   - Applied 99.9th percentile threshold for extreme values

3. **Data Validation:**
   - Filtered invalid stock codes (non-product entries)
   - Ensured date consistency
   - Validated revenue calculations

### 1.4 Feature Engineering
```python
# Created features
- Revenue = Quantity × UnitPrice
- Date components (Year, Month, Quarter, Weekday)
- Invoice-level aggregations
- Product categories (keyword-based)
- Customer frequency indicators
```

## 2. Customer Segmentation Methodology

### 2.1 RFM Analysis
**RFM Framework:**
- **Recency (R):** Days since last purchase
- **Frequency (F):** Number of unique orders
- **Monetary (M):** Total revenue generated

**Calculation:**
```python
analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
rfm_data = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (analysis_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Revenue': 'sum'
})
```

**Scoring System:**
- Quintile-based scoring (1-5 scale)
- Recency: Lower values = Higher scores
- Frequency & Monetary: Higher values = Higher scores

### 2.2 Traditional RFM Segmentation
**Segment Definitions:**
- **Champions:** R≥4, F≥4, M≥4
- **Loyal Customers:** F≥4, M≥4
- **Potential Loyalists:** R≥4, M≥3
- **New Customers:** R≥4, F≤2
- **At Risk:** R≤2, M≥3
- **Lost:** Low across all dimensions

### 2.3 Machine Learning Clustering
**Algorithm:** K-Means Clustering

**Features Used:**
```python
clustering_features = [
    'Recency', 'Frequency', 'Monetary',
    'AvgOrderValue', 'UniqueProducts'
]
```

**Optimization Process:**
1. Feature standardization using StandardScaler
2. Elbow method for optimal cluster count
3. Silhouette score validation
4. PCA for dimensionality reduction and visualization

**Cluster Naming:**
- Based on RFM characteristics
- Business-meaningful labels
- Revenue contribution analysis

## 3. Market Trends Analysis

### 3.1 Revenue Trend Analysis
**Temporal Aggregations:**
- Daily revenue patterns
- Weekly seasonality
- Monthly trends with growth rates
- Quarterly performance

**Statistical Methods:**
```python
# Growth rate calculation
monthly_revenue['GrowthRate'] = monthly_revenue['Revenue'].pct_change() * 100

# Correlation analysis
correlation, p_value = pearsonr(time_index, revenue_values)
```

### 3.2 Seasonal Pattern Detection
**Analysis Dimensions:**
- Monthly seasonality indices
- Day-of-week patterns
- Holiday period analysis (Christmas, etc.)
- Quarterly performance comparison

**Seasonality Index:**
```python
seasonality_index = (monthly_revenue / average_monthly_revenue) * 100
```

**Statistical Validation:**
- ANOVA testing for seasonal differences
- F-statistic calculation for significance

### 3.3 Product Performance Analysis
**ABC Classification:**
- **A Products:** Top 20% by revenue (80% of total revenue)
- **B Products:** Next 30% by revenue (15% of total revenue)
- **C Products:** Remaining 50% by revenue (5% of total revenue)

**Category Analysis:**
- Keyword-based categorization
- Cross-category performance comparison
- Customer reach analysis by category

**Concentration Analysis:**
- Pareto principle validation (80/20 rule)
- Revenue distribution analysis
- Product lifecycle assessment

## 4. Statistical Validation

### 4.1 Hypothesis Testing
**Customer Segmentation:**
```python
# ANOVA test for segment differences
f_statistic, p_value = f_oneway(*cluster_groups)
# H0: No difference between segments
# H1: Significant difference exists
```

**Seasonal Patterns:**
```python
# Test for seasonal effects
monthly_groups = [df[df['Month'] == m]['Revenue'] for m in range(1, 13)]
f_stat, p_value = stats.f_oneway(*monthly_groups)
```

### 4.2 Model Quality Metrics
**Clustering Validation:**
- Silhouette Score: Measures cluster cohesion and separation
- Inertia: Within-cluster sum of squares
- Calinski-Harabasz Index: Ratio of between/within cluster dispersion

**Significance Thresholds:**
- α = 0.05 for all statistical tests
- Confidence interval: 95%
- Effect size consideration for practical significance

## 5. Visualization Strategy

### 5.1 Chart Types and Purpose
**Customer Segmentation:**
- 3D scatter plots for RFM visualization
- Pie charts for segment distribution
- Bar charts for segment performance comparison

**Revenue Trends:**
- Line plots for temporal trends
- Bar charts for growth rates
- Heatmaps for seasonal patterns

**Product Analysis:**
- Horizontal bar charts for top products
- Pareto charts for concentration analysis
- Treemaps for category performance

### 5.2 Color Schemes and Accessibility
- Colorblind-friendly palettes
- High contrast for readability
- Consistent color mapping across visualizations

## 6. Technology Stack

### 6.1 Core Libraries
```python
# Data Processing
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Statistical Analysis
import scipy.stats as stats
from scipy.stats import f_oneway, pearsonr

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
```

### 6.2 Environment Management
```bash
# Conda environment
conda create -n online_retail_analysis python=3.9
conda activate online_retail_analysis
pip install -r requirements.txt
```

## 7. Quality Assurance

### 7.1 Data Validation Checks
```python
def validate_processed_data(df):
    validations = {
        'No missing CustomerID': df['CustomerID'].isnull().sum() == 0,
        'Positive quantities': (df['Quantity'] > 0).all(),
        'Positive prices': (df['UnitPrice'] > 0).all(),
        'Valid dates': pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']),
        'Revenue calculated': 'Revenue' in df.columns
    }
    return all(validations.values())
```

### 7.2 Reproducibility Measures
- Fixed random seeds for clustering
- Version-controlled code repository
- Documented parameter choices
- Comprehensive logging

## 8. Limitations and Assumptions

### 8.1 Data Limitations
- **Temporal Scope:** 13-month period may not capture full seasonal cycles
- **Geographic Scope:** Primarily UK-based, limited international insights
- **Product Categories:** Simplified keyword-based categorization
- **Customer Demographics:** No age, gender, or income data available

### 8.2 Methodological Assumptions
- **RFM Validity:** Assumes RFM metrics are primary drivers of customer value
- **Clustering Assumption:** Assumes spherical clusters (K-means limitation)
- **Seasonality:** Assumes consistent seasonal patterns year-over-year
- **Stationarity:** Assumes underlying business model remains constant

### 8.3 Statistical Assumptions
- **Normality:** Some tests assume normal distribution of residuals
- **Independence:** Assumes transactions are independent events
- **Homoscedasticity:** Assumes constant variance in error terms

## 9. Future Enhancements

### 9.1 Data Enrichment
- Customer demographic integration
- Competitive pricing data
- Marketing campaign data
- External economic indicators

### 9.2 Advanced Analytics
- Predictive modeling (customer lifetime value)
- Recommendation systems
- Real-time analysis pipeline
- A/B testing framework

### 9.3 Technical Improvements
- Automated data pipeline
- Interactive dashboard with real-time updates
- API integration for live data feeds
- Cloud deployment for scalability

---

## References

1. Hughes, A. M. (2005). Strategic Database Marketing. McGraw-Hill.
2. Wedel, M., & Kamakura, W. A. (2000). Market Segmentation: Conceptual and Methodological Foundations. Springer.
3. UCI Machine Learning Repository: Online Retail Dataset
4. Scikit-learn Documentation: Clustering Algorithms
5. Plotly Documentation: Interactive Visualizations

---

**Document Version:** 1.0  
**Last Updated:** July 2025  
**Next Review:** December 2025
