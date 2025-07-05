"""
Customer Segmentation Analysis Module

Performs RFM (Recency, Frequency, Monetary) analysis and advanced customer segmentation
using clustering algorithms. Identifies distinct customer behavior patterns.

Author: Bryant M.
Date: July 2025
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import scipy.stats as stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class CustomerSegmentationAnalyst:
    """
    Advanced customer segmentation using RFM analysis and clustering
    """
    
    def __init__(self, config):
        """Initialize segmentation analyst with configuration"""
        self.config = config
        self.analysis_config = config['analysis']
        self.output_path = config['output']['reports_path']
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def calculate_rfm_metrics(self, df):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            pd.DataFrame: RFM metrics per customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Define analysis date (latest date + 1 day)
        analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
        logger.info(f"Analysis date set to: {analysis_date}")
        
        # Calculate RFM metrics
        rfm_data = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (analysis_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'Revenue': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Additional customer metrics
        customer_metrics = df.groupby('CustomerID').agg({
            'Quantity': ['sum', 'mean'],
            'UnitPrice': 'mean',
            'StockCode': 'nunique',
            'InvoiceDate': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_metrics.columns = [
            'CustomerID', 'TotalQuantity', 'AvgQuantityPerOrder', 
            'AvgUnitPrice', 'UniqueProducts', 'FirstPurchase', 'LastPurchase'
        ]
        
        # Calculate customer lifetime (days)
        customer_metrics['CustomerLifetime'] = (
            customer_metrics['LastPurchase'] - customer_metrics['FirstPurchase']
        ).dt.days
        
        # Merge RFM with additional metrics
        rfm_enhanced = rfm_data.merge(customer_metrics, on='CustomerID', how='left')
        
        # Calculate derived metrics
        rfm_enhanced['AvgOrderValue'] = rfm_enhanced['Monetary'] / rfm_enhanced['Frequency']
        rfm_enhanced['PurchaseFrequency'] = rfm_enhanced['Frequency'] / (rfm_enhanced['CustomerLifetime'] + 1)
        
        logger.info(f"[OK] RFM metrics calculated for {len(rfm_enhanced)} customers")
        logger.info(f"[OK] Recency range: {rfm_enhanced['Recency'].min():.0f} to {rfm_enhanced['Recency'].max():.0f} days")
        logger.info(f"[OK] Frequency range: {rfm_enhanced['Frequency'].min():.0f} to {rfm_enhanced['Frequency'].max():.0f} orders")
        logger.info(f"[OK] Monetary range: £{rfm_enhanced['Monetary'].min():.2f} to £{rfm_enhanced['Monetary'].max():.2f}")
        
        return rfm_enhanced
    
    def create_rfm_segments(self, rfm_data):
        """
        Create traditional RFM segments using quintile scoring
        
        Args:
            rfm_data (pd.DataFrame): RFM metrics data
            
        Returns:
            pd.DataFrame: RFM data with quintile scores and segments
        """
        logger.info("Creating RFM quintile segments...")
        
        # Create RFM scores (1-5, where 5 is best)
        # Note: For Recency, lower values are better, so we reverse the scoring
        rfm_scored = rfm_data.copy()
        
        # Recency scoring (1 = most recent, 5 = least recent) - REVERSED for intuitive scoring
        rfm_scored['R_Score'] = pd.qcut(rfm_scored['Recency'], q=5, labels=[5,4,3,2,1])
        
        # Frequency scoring (5 = highest frequency)
        rfm_scored['F_Score'] = pd.qcut(rfm_scored['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5])
        
        # Monetary scoring (5 = highest monetary value)
        rfm_scored['M_Score'] = pd.qcut(rfm_scored['Monetary'], q=5, labels=[1,2,3,4,5])
        
        # Convert to numeric
        rfm_scored['R_Score'] = rfm_scored['R_Score'].astype(int)
        rfm_scored['F_Score'] = rfm_scored['F_Score'].astype(int)
        rfm_scored['M_Score'] = rfm_scored['M_Score'].astype(int)
        
        # Create RFM combined score
        rfm_scored['RFM_Score'] = (
            rfm_scored['R_Score'].astype(str) + 
            rfm_scored['F_Score'].astype(str) + 
            rfm_scored['M_Score'].astype(str)
        )
        
        # Create customer segments based on RFM scores
        rfm_scored['Segment'] = rfm_scored.apply(self._assign_rfm_segment, axis=1)
        
        logger.info("[OK] RFM quintile segments created")
        
        return rfm_scored
    
    def _assign_rfm_segment(self, row):
        """
        Assign customer segment based on RFM scores
        
        Args:
            row: DataFrame row with R_Score, F_Score, M_Score
            
        Returns:
            str: Customer segment name
        """
        R, F, M = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: High value, frequent, recent customers
        if R >= 4 and F >= 4 and M >= 4:
            return 'Champions'
        
        # Loyal Customers: High frequency and monetary, but not necessarily recent
        elif F >= 4 and M >= 4:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with good monetary value
        elif R >= 4 and M >= 3:
            return 'Potential Loyalists'
        
        # New Customers: Recent but low frequency/monetary
        elif R >= 4 and F <= 2:
            return 'New Customers'
        
        # Promising: Recent with moderate frequency
        elif R >= 3 and F >= 2 and M >= 2:
            return 'Promising'
        
        # Customers Needing Attention: Above average recency, frequency & monetary
        elif R >= 3 and F >= 3 and M >= 3:
            return 'Customers Needing Attention'
        
        # About to Sleep: Below average recency, frequency & monetary
        elif R <= 2 and F <= 2:
            return 'About to Sleep'
        
        # At Risk: Good monetary value but haven't purchased recently
        elif R <= 2 and M >= 3:
            return 'At Risk'
        
        # Cannot Lose Them: High monetary value but low recency and frequency
        elif F <= 2 and M >= 4:
            return 'Cannot Lose Them'
        
        # Hibernating: Low recency, frequency & monetary
        elif R <= 2 and F <= 2 and M <= 2:
            return 'Hibernating'
        
        # Lost: Lowest recency, frequency & monetary
        else:
            return 'Lost'
    
    def perform_clustering_analysis(self, rfm_data):
        """
        Perform K-means clustering on RFM data to identify customer segments
        
        Args:
            rfm_data (pd.DataFrame): RFM metrics data
            
        Returns:
            dict: Clustering results and analysis
        """
        logger.info("Performing K-means clustering analysis...")
        
        # Prepare data for clustering
        clustering_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'UniqueProducts']
        X = rfm_data[clustering_features].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters using elbow method and silhouette score
        cluster_range = range(2, 9)
        inertias = []
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Find optimal number of clusters (highest silhouette score)
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        logger.info(f"[OK] Optimal number of clusters: {optimal_clusters}")
        logger.info(f"[OK] Best silhouette score: {max(silhouette_scores):.3f}")
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to RFM data
        rfm_clustered = rfm_data.copy()
        rfm_clustered['Cluster'] = cluster_labels
        rfm_clustered['Cluster_Name'] = rfm_clustered['Cluster'].map(
            self._name_clusters(rfm_clustered, optimal_clusters)
        )
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        rfm_clustered['PCA1'] = X_pca[:, 0]
        rfm_clustered['PCA2'] = X_pca[:, 1]
        
        clustering_results = {
            'rfm_clustered': rfm_clustered,
            'cluster_centers': final_kmeans.cluster_centers_,
            'feature_names': clustering_features,
            'scaler': scaler,
            'pca': pca,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_clusters': optimal_clusters,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
        
        logger.info("[OK] Clustering analysis completed")
        
        return clustering_results
    
    def _name_clusters(self, rfm_clustered, n_clusters):
        """
        Assign meaningful names to clusters based on their characteristics
        
        Args:
            rfm_clustered (pd.DataFrame): RFM data with cluster assignments
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Mapping of cluster numbers to names
        """
        cluster_profiles = rfm_clustered.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'AvgOrderValue': 'mean'
        })
        
        # Rank clusters by different metrics
        recency_rank = cluster_profiles['Recency'].rank(ascending=True)  # Lower recency is better
        frequency_rank = cluster_profiles['Frequency'].rank(ascending=False)  # Higher frequency is better
        monetary_rank = cluster_profiles['Monetary'].rank(ascending=False)  # Higher monetary is better
        
        # Combined ranking (lower is better)
        combined_rank = recency_rank + frequency_rank + monetary_rank
        
        # Assign names based on rankings and characteristics
        cluster_names = {}
        sorted_clusters = combined_rank.sort_values().index.tolist()
        
        names = ['VIP Champions', 'Loyal Customers', 'Promising Customers', 'Regular Customers', 
                'At-Risk Customers', 'Lost Customers', 'Occasional Buyers', 'Budget Conscious']
        
        for i, cluster in enumerate(sorted_clusters):
            cluster_names[cluster] = names[min(i, len(names)-1)]
        
        return cluster_names
    
    def analyze_segments(self, rfm_scored, clustering_results):
        """
        Comprehensive analysis of customer segments
        
        Args:
            rfm_scored (pd.DataFrame): RFM data with traditional segments
            clustering_results (dict): Clustering analysis results
            
        Returns:
            dict: Segment analysis results
        """
        logger.info("Analyzing customer segments...")
        
        rfm_clustered = clustering_results['rfm_clustered']
        
        # Traditional RFM segment analysis
        rfm_segment_analysis = rfm_scored.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum'],
            'AvgOrderValue': ['mean', 'median'],
            'UniqueProducts': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        rfm_segment_analysis.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in rfm_segment_analysis.columns]
        rfm_segment_analysis = rfm_segment_analysis.rename(columns={'CustomerID_count': 'Customer_Count'})
        
        # Calculate percentages
        total_customers = rfm_scored['CustomerID'].nunique()
        total_revenue = rfm_scored['Monetary'].sum()
        
        rfm_segment_analysis['Customer_Percentage'] = (rfm_segment_analysis['Customer_Count'] / total_customers * 100).round(2)
        rfm_segment_analysis['Revenue_Percentage'] = (rfm_segment_analysis['Monetary_sum'] / total_revenue * 100).round(2)
        
        # Clustering-based segment analysis
        cluster_segment_analysis = rfm_clustered.groupby('Cluster_Name').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum'],
            'AvgOrderValue': ['mean', 'median'],
            'UniqueProducts': ['mean', 'median']
        }).round(2)
        
        # Flatten column names
        cluster_segment_analysis.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in cluster_segment_analysis.columns]
        cluster_segment_analysis = cluster_segment_analysis.rename(columns={'CustomerID_count': 'Customer_Count'})
        
        # Calculate percentages
        cluster_segment_analysis['Customer_Percentage'] = (cluster_segment_analysis['Customer_Count'] / total_customers * 100).round(2)
        cluster_segment_analysis['Revenue_Percentage'] = (cluster_segment_analysis['Monetary_sum'] / total_revenue * 100).round(2)
        
        # Statistical significance tests between segments
        segment_comparison = self._perform_segment_statistical_tests(rfm_clustered)
        
        segment_analysis = {
            'rfm_traditional_segments': rfm_segment_analysis,
            'clustering_segments': cluster_segment_analysis,
            'statistical_tests': segment_comparison,
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'segment_summary': {
                'traditional_segments_count': len(rfm_segment_analysis),
                'cluster_segments_count': len(cluster_segment_analysis),
                'top_revenue_segment_traditional': rfm_segment_analysis['Revenue_Percentage'].idxmax(),
                'top_revenue_segment_cluster': cluster_segment_analysis['Revenue_Percentage'].idxmax()
            }
        }
        
        logger.info("[OK] Segment analysis completed")
        logger.info(f"[OK] Traditional segments: {len(rfm_segment_analysis)}")
        logger.info(f"[OK] Cluster-based segments: {len(cluster_segment_analysis)}")
        
        return segment_analysis
    
    def _perform_segment_statistical_tests(self, rfm_clustered):
        """
        Perform statistical tests to validate segment differences
        
        Args:
            rfm_clustered (pd.DataFrame): RFM data with cluster assignments
            
        Returns:
            dict: Statistical test results
        """
        from scipy.stats import f_oneway, kruskal
        
        # Group data by clusters
        cluster_groups = [group['Monetary'].values for name, group in rfm_clustered.groupby('Cluster')]
        
        # ANOVA test for monetary differences
        f_stat, p_value_anova = f_oneway(*cluster_groups)
        
        # Kruskal-Wallis test (non-parametric alternative)
        h_stat, p_value_kruskal = kruskal(*cluster_groups)
        
        return {
            'anova_f_statistic': f_stat,
            'anova_p_value': p_value_anova,
            'kruskal_h_statistic': h_stat,
            'kruskal_p_value': p_value_kruskal,
            'segments_significantly_different': p_value_anova < 0.05
        }
    
    def save_segmentation_results(self, rfm_scored, clustering_results, segment_analysis):
        """
        Save all segmentation results to files
        
        Args:
            rfm_scored (pd.DataFrame): RFM data with traditional segments
            clustering_results (dict): Clustering analysis results
            segment_analysis (dict): Segment analysis results
        """
        output_dir = Path(self.config['data']['processed_data_path'])
        
        # Save RFM data with segments
        rfm_scored.to_csv(output_dir / 'customer_rfm_segments.csv', index=False)
        logger.info(f"[OK] RFM segments saved to {output_dir / 'customer_rfm_segments.csv'}")
        
        # Save clustering results
        clustering_results['rfm_clustered'].to_csv(output_dir / 'customer_cluster_segments.csv', index=False)
        logger.info(f"[OK] Cluster segments saved to {output_dir / 'customer_cluster_segments.csv'}")
        
        # Save segment summaries
        segment_analysis['rfm_traditional_segments'].to_csv(output_dir / 'rfm_segment_summary.csv')
        segment_analysis['clustering_segments'].to_csv(output_dir / 'cluster_segment_summary.csv')
        
        # Save detailed analysis report
        report_path = output_dir / 'segmentation_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("CUSTOMER SEGMENTATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Customers Analyzed: {segment_analysis['total_customers']:,}\n")
            f.write(f"Total Revenue: £{segment_analysis['total_revenue']:,.2f}\n")
            f.write(f"Traditional RFM Segments: {segment_analysis['segment_summary']['traditional_segments_count']}\n")
            f.write(f"Cluster-based Segments: {segment_analysis['segment_summary']['cluster_segments_count']}\n\n")
            
            f.write("STATISTICAL VALIDATION\n")
            f.write("-" * 20 + "\n")
            stats = segment_analysis['statistical_tests']
            f.write(f"ANOVA F-statistic: {stats['anova_f_statistic']:.4f}\n")
            f.write(f"ANOVA p-value: {stats['anova_p_value']:.6f}\n")
            f.write(f"Segments significantly different: {stats['segments_significantly_different']}\n\n")
        
        logger.info(f"[OK] Segmentation report saved to {report_path}")
    
    def perform_segmentation(self, df):
        """
        Main segmentation pipeline
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Complete segmentation results
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING CUSTOMER SEGMENTATION ANALYSIS")
            logger.info("=" * 50)
            
            # Step 1: Calculate RFM metrics
            rfm_data = self.calculate_rfm_metrics(df)
            
            # Step 2: Create traditional RFM segments
            rfm_scored = self.create_rfm_segments(rfm_data)
            
            # Step 3: Perform clustering analysis
            clustering_results = self.perform_clustering_analysis(rfm_data)
            
            # Step 4: Analyze segments
            segment_analysis = self.analyze_segments(rfm_scored, clustering_results)
            
            # Step 5: Save results
            self.save_segmentation_results(rfm_scored, clustering_results, segment_analysis)
            
            # Compile final results
            final_results = {
                'rfm_data': rfm_data,
                'rfm_scored': rfm_scored,
                'clustering_results': clustering_results,
                'segment_analysis': segment_analysis
            }
            
            logger.info("=" * 50)
            logger.info("CUSTOMER SEGMENTATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Customer segmentation failed: {str(e)}")
            raise