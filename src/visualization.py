#!/usr/bin/env python3
"""
Visualization Engine Module

Creates comprehensive visualizations for e-commerce retail analysis including
customer segmentation charts, revenue trends, seasonal patterns, and interactive dashboards.

Author: Bryant M.
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import logging
from pathlib import Path
import warnings
from datetime import datetime
import calendar

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RetailVisualizationEngine:
    """
    Comprehensive visualization engine for retail analysis
    """
    
    def __init__(self, config):
        """Initialize visualization engine"""
        self.config = config
        self.output_path = Path(config['output']['visualizations_path'])
        self.dashboard_path = Path(config['output']['dashboards_path'])
        
        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.dashboard_path.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting parameters
        self.figure_size = (12, 8)
        self.color_palette = sns.color_palette("husl", 10)
        
    def create_customer_segmentation_visualizations(self, segmentation_results):
        """
        Create visualizations for customer segmentation analysis
        
        Args:
            segmentation_results (dict): Segmentation analysis results
        """
        logger.info("Creating customer segmentation visualizations...")
        
        rfm_clustered = segmentation_results['clustering_results']['rfm_clustered']
        segment_analysis = segmentation_results['segment_analysis']
        
        # 1. RFM Segments Distribution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segmentation Analysis', fontsize=16, fontweight='bold')
        
        # Traditional RFM segments
        rfm_segments = segmentation_results['rfm_scored']['Segment'].value_counts()
        colors1 = plt.cm.Set3(np.linspace(0, 1, len(rfm_segments)))
        
        ax1.pie(rfm_segments.values, labels=rfm_segments.index, autopct='%1.1f%%', 
                colors=colors1, startangle=90)
        ax1.set_title('Traditional RFM Segments Distribution', fontweight='bold')
        
        # Cluster-based segments
        cluster_segments = rfm_clustered['Cluster_Name'].value_counts()
        colors2 = plt.cm.Set2(np.linspace(0, 1, len(cluster_segments)))
        
        ax2.pie(cluster_segments.values, labels=cluster_segments.index, autopct='%1.1f%%',
                colors=colors2, startangle=90)
        ax2.set_title('ML Cluster-Based Segments', fontweight='bold')
        
        # RFM Score Distribution
        rfm_scored = segmentation_results['rfm_scored']
        ax3.hist([rfm_scored['R_Score'], rfm_scored['F_Score'], rfm_scored['M_Score']], 
                bins=5, alpha=0.7, label=['Recency', 'Frequency', 'Monetary'], 
                color=['red', 'green', 'blue'])
        ax3.set_xlabel('RFM Scores')
        ax3.set_ylabel('Number of Customers')
        ax3.set_title('RFM Scores Distribution', fontweight='bold')
        ax3.legend()
        
        # Customer Value vs Frequency Scatter
        scatter_colors = [self.color_palette[i] for i in rfm_clustered['Cluster']]
        scatter = ax4.scatter(rfm_clustered['Frequency'], rfm_clustered['Monetary'], 
                             c=scatter_colors, alpha=0.6, s=50)
        ax4.set_xlabel('Purchase Frequency')
        ax4.set_ylabel('Monetary Value (£)')
        ax4.set_title('Customer Value vs Frequency by Segment', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'customer_segments_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. RFM 3D Scatter Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for cluster in rfm_clustered['Cluster'].unique():
            cluster_data = rfm_clustered[rfm_clustered['Cluster'] == cluster]
            ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                      cluster_data['Monetary'], label=f'Cluster {cluster}', alpha=0.6)
        
        ax.set_xlabel('Recency (Days)')
        ax.set_ylabel('Frequency (Orders)')
        ax.set_zlabel('Monetary Value (£)')
        ax.set_title('3D Customer Segmentation - RFM Analysis', fontweight='bold')
        ax.legend()
        
        plt.savefig(self.output_path / 'rfm_3d_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Segment Performance Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Revenue by segment
        cluster_revenue = segment_analysis['clustering_segments']['Monetary_sum'].sort_values(ascending=True)
        cluster_revenue.plot(kind='barh', ax=ax1, color=self.color_palette[:len(cluster_revenue)])
        ax1.set_title('Total Revenue by Customer Segment', fontweight='bold')
        ax1.set_xlabel('Total Revenue (£)')
        
        # Customer count by segment
        cluster_count = segment_analysis['clustering_segments']['Customer_Count'].sort_values(ascending=True)
        cluster_count.plot(kind='barh', ax=ax2, color=self.color_palette[:len(cluster_count)])
        ax2.set_title('Customer Count by Segment', fontweight='bold')
        ax2.set_xlabel('Number of Customers')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'segment_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Customer segmentation visualizations created")
    
    def create_revenue_trend_visualizations(self, market_insights):
        """
        Create revenue trend visualizations
        
        Args:
            market_insights (dict): Market analysis results
        """
        logger.info("Creating revenue trend visualizations...")
        
        trend_analysis = market_insights['trend_analysis']
        
        # 1. Monthly Revenue Trends
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Revenue Trends Analysis', fontsize=16, fontweight='bold')
        
        # Monthly revenue trend
        monthly_data = trend_analysis['monthly_revenue']
        ax1.plot(monthly_data['YearMonth'].astype(str), monthly_data['MonthlyRevenue'], 
                marker='o', linewidth=2, markersize=6, color='blue')
        ax1.set_title('Monthly Revenue Trend', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Revenue (£)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Monthly growth rate
        ax2.bar(monthly_data['YearMonth'].astype(str), monthly_data['RevenueGrowthRate'], 
               color=['green' if x > 0 else 'red' for x in monthly_data['RevenueGrowthRate']])
        ax2.set_title('Monthly Revenue Growth Rate', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Growth Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Daily revenue trend
        daily_data = trend_analysis['daily_revenue']
        ax3.plot(daily_data['Date'], daily_data['DailyRevenue'], alpha=0.7, color='purple')
        ax3.set_title('Daily Revenue Trend', fontweight='bold')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Revenue (£)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Average order value trend
        ax4.plot(monthly_data['YearMonth'].astype(str), monthly_data['MonthlyAvgOrderValue'], 
                marker='s', linewidth=2, markersize=6, color='orange')
        ax4.set_title('Monthly Average Order Value', fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Order Value (£)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'revenue_trends_monthly.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Weekly Revenue Pattern
        fig, ax = plt.subplots(figsize=(12, 6))
        weekly_data = trend_analysis['weekly_revenue']
        ax.plot(weekly_data['WeekStart'], weekly_data['WeeklyRevenue'], 
               marker='o', linewidth=2, alpha=0.8, color='green')
        ax.set_title('Weekly Revenue Pattern', fontweight='bold', fontsize=14)
        ax.set_xlabel('Week')
        ax.set_ylabel('Weekly Revenue (£)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'weekly_revenue_pattern.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Revenue trend visualizations created")
    
    def create_seasonal_analysis_visualizations(self, market_insights):
        """
        Create seasonal analysis visualizations
        
        Args:
            market_insights (dict): Market analysis results
        """
        logger.info("Creating seasonal analysis visualizations...")
        
        seasonal_analysis = market_insights['seasonal_analysis']
        
        # 1. Comprehensive Seasonal Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Seasonal Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Monthly seasonality
        monthly_patterns = seasonal_analysis['monthly_patterns']
        months = [calendar.month_name[i] for i in monthly_patterns.index]
        
        bars1 = ax1.bar(months, monthly_patterns['TotalRevenue'], 
                       color=self.color_palette[:len(months)])
        ax1.set_title('Revenue by Month', fontweight='bold')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Revenue (£)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'£{height:,.0f}', ha='center', va='bottom', fontsize=8)
        
        # Seasonality index
        ax2.plot(months, monthly_patterns['SeasonalityIndex'], 
                marker='o', linewidth=3, markersize=8, color='red')
        ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Average (100)')
        ax2.set_title('Seasonality Index by Month', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Seasonality Index')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Day of week patterns
        dow_patterns = seasonal_analysis['day_of_week_patterns']
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Ensure we have data for all 7 days (fill missing days with 0)
        dow_data = []
        for i in range(7):
            if i in dow_patterns.index:
                dow_data.append(dow_patterns.loc[i, 'TotalRevenue'])
            else:
                dow_data.append(0)
        
        bars3 = ax3.bar(day_names, dow_data, color=self.color_palette[:7])
        ax3.set_title('Revenue by Day of Week', fontweight='bold')
        ax3.set_xlabel('Day of Week')
        ax3.set_ylabel('Total Revenue (£)')
        
        # Quarterly patterns
        quarterly_patterns = seasonal_analysis['quarterly_patterns']
        
        # Ensure we have data for all 4 quarters
        quarter_data = []
        quarter_names = []
        for i in range(1, 5):
            if i in quarterly_patterns.index:
                quarter_data.append(quarterly_patterns.loc[i, 'TotalRevenue'])
                quarter_names.append(f'Q{i}')
            else:
                quarter_data.append(0)
                quarter_names.append(f'Q{i}')
        
        colors_quarterly = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'][:len(quarter_data)]
        bars4 = ax4.bar(quarter_names, quarter_data, color=colors_quarterly)
        ax4.set_title('Revenue by Quarter', fontweight='bold')
        ax4.set_xlabel('Quarter')
        ax4.set_ylabel('Total Revenue (£)')
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'£{height:,.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Holiday Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Christmas period analysis
        holiday_analysis = seasonal_analysis['holiday_analysis']
        
        holiday_data = {
            'December Only': holiday_analysis['december_share'],
            'Nov-Dec Combined': holiday_analysis['nov_dec_share'],
            'Rest of Year': 100 - holiday_analysis['nov_dec_share']
        }
        
        colors = ['red', 'orange', 'lightblue']
        wedges, texts, autotexts = ax1.pie(holiday_data.values(), labels=holiday_data.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Holiday Season Revenue Share', fontweight='bold')
        
        # Monthly customer count
        monthly_customers = []
        for i in range(1, 13):
            if i in monthly_patterns.index:
                monthly_customers.append(monthly_patterns.loc[i, 'UniqueCustomers'])
            else:
                monthly_customers.append(0)
        
        ax2.bar(months, monthly_customers, 
               color=self.color_palette[:len(months)], alpha=0.7)
        ax2.set_title('Unique Customers by Month', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Unique Customers')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'holiday_seasonal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Seasonal analysis visualizations created")
    
    def create_product_performance_visualizations(self, market_insights):
        """
        Create product performance visualizations
        
        Args:
            market_insights (dict): Market analysis results
        """
        logger.info("Creating product performance visualizations...")
        
        product_analysis = market_insights['product_analysis']
        
        # 1. Product Performance Overview
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Product Performance Analysis', fontsize=16, fontweight='bold')
        
        # Top 10 products by revenue
        top_products = product_analysis['top_products']['top_10_by_revenue']
        product_names = [desc[:30] + '...' if len(desc) > 30 else desc 
                        for desc in top_products['Description'].head(10)]
        
        bars1 = ax1.barh(range(10), top_products['TotalRevenue'].head(10), 
                        color=self.color_palette[:10])
        ax1.set_yticks(range(10))
        ax1.set_yticklabels(product_names, fontsize=8)
        ax1.set_title('Top 10 Products by Revenue', fontweight='bold')
        ax1.set_xlabel('Total Revenue (£)')
        
        # Category performance
        category_performance = product_analysis['category_performance']
        bars2 = ax2.bar(category_performance.index, category_performance['TotalRevenue'], 
                       color=self.color_palette[:len(category_performance)])
        ax2.set_title('Revenue by Product Category', fontweight='bold')
        ax2.set_xlabel('Product Category')
        ax2.set_ylabel('Total Revenue (£)')
        ax2.tick_params(axis='x', rotation=45)
        
        # ABC Analysis
        abc_analysis = product_analysis['concentration_analysis']['abc_analysis']
        colors_abc = ['gold', 'silver', 'lightcoral']
        bars3 = ax3.bar(abc_analysis.index, abc_analysis.values, color=colors_abc)
        ax3.set_title('ABC Analysis - Product Classification', fontweight='bold')
        ax3.set_xlabel('ABC Category')
        ax3.set_ylabel('Number of Products')
        
        # Add percentage labels
        total_products = abc_analysis.sum()
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            percentage = height / total_products * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # Revenue concentration (Pareto analysis)
        product_performance = product_analysis['product_performance']
        cumulative_revenue_pct = product_performance['CumulativeRevenueShare'].values
        product_rank = range(1, len(cumulative_revenue_pct) + 1)
        
        ax4.plot(product_rank, cumulative_revenue_pct, color='blue', linewidth=2)
        ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% of Revenue')
        ax4.fill_between(product_rank, cumulative_revenue_pct, alpha=0.3, color='blue')
        ax4.set_title('Revenue Concentration (Pareto Analysis)', fontweight='bold')
        ax4.set_xlabel('Product Rank')
        ax4.set_ylabel('Cumulative Revenue %')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'product_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Product Category Deep Dive
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Category revenue share pie chart
        category_revenue_share = category_performance['RevenueShare']
        colors_cat = plt.cm.Set3(np.linspace(0, 1, len(category_revenue_share)))
        
        wedges, texts, autotexts = ax1.pie(category_revenue_share.values, 
                                          labels=category_revenue_share.index,
                                          autopct='%1.1f%%', colors=colors_cat, startangle=90)
        ax1.set_title('Revenue Share by Product Category', fontweight='bold')
        
        # Category customer reach
        ax2.bar(category_performance.index, category_performance['UniqueCustomers'], 
               color=self.color_palette[:len(category_performance)], alpha=0.7)
        ax2.set_title('Customer Reach by Category', fontweight='bold')
        ax2.set_xlabel('Product Category')
        ax2.set_ylabel('Number of Unique Customers')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'category_deep_dive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Product performance visualizations created")
    
    def create_geographic_visualizations(self, market_insights):
        """
        Create geographic analysis visualizations
        
        Args:
            market_insights (dict): Market analysis results
        """
        logger.info("Creating geographic visualizations...")
        
        geographic_analysis = market_insights['geographic_analysis']
        
        if not geographic_analysis.get('geographic_data_available', False):
            logger.info("Geographic data not available, skipping geographic visualizations")
            return
        
        # 1. Country Performance Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Geographic Market Analysis', fontsize=16, fontweight='bold')
        
        country_performance = geographic_analysis['country_performance']
        top_countries = country_performance.head(10)
        
        # Top countries by revenue
        bars1 = ax1.barh(range(len(top_countries)), top_countries['TotalRevenue'], 
                        color=self.color_palette[:len(top_countries)])
        ax1.set_yticks(range(len(top_countries)))
        ax1.set_yticklabels(top_countries.index, fontsize=10)
        ax1.set_title('Top 10 Countries by Revenue', fontweight='bold')
        ax1.set_xlabel('Total Revenue (£)')
        
        # Revenue share pie chart
        top_5_countries = country_performance.head(5)
        other_revenue = country_performance['TotalRevenue'][5:].sum()
        
        pie_data = list(top_5_countries['TotalRevenue']) + [other_revenue]
        pie_labels = list(top_5_countries.index) + ['Others']
        
        colors_geo = plt.cm.Set2(np.linspace(0, 1, len(pie_data)))
        ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', colors=colors_geo, startangle=90)
        ax2.set_title('Revenue Distribution by Country', fontweight='bold')
        
        # Average revenue per customer by country
        ax3.bar(top_countries.index, top_countries['AvgRevenuePerCustomer'], 
               color=self.color_palette[:len(top_countries)], alpha=0.7)
        ax3.set_title('Avg Revenue per Customer by Country', fontweight='bold')
        ax3.set_xlabel('Country')
        ax3.set_ylabel('Avg Revenue per Customer (£)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Number of customers by country
        ax4.bar(top_countries.index, top_countries['UniqueCustomers'], 
               color=self.color_palette[:len(top_countries)], alpha=0.7)
        ax4.set_title('Number of Customers by Country', fontweight='bold')
        ax4.set_xlabel('Country')
        ax4.set_ylabel('Number of Customers')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Geographic visualizations created")
    
    def create_interactive_dashboard(self, cleaned_data, segmentation_results, market_insights):
        """
        Create interactive Plotly dashboard
        
        Args:
            cleaned_data (pd.DataFrame): Cleaned transaction data
            segmentation_results (dict): Segmentation analysis results
            market_insights (dict): Market analysis results
        """
        logger.info("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Monthly Revenue Trend', 'Customer Segments Distribution',
                          'Product Category Performance', 'Seasonal Patterns',
                          'Top Products by Revenue', 'Geographic Distribution'),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # 1. Monthly Revenue Trend
        monthly_data = market_insights['trend_analysis']['monthly_revenue']
        fig.add_trace(
            go.Scatter(x=monthly_data['YearMonth'].astype(str), 
                      y=monthly_data['MonthlyRevenue'],
                      mode='lines+markers', name='Monthly Revenue',
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # 2. Customer Segments Pie Chart
        cluster_segments = segmentation_results['clustering_results']['rfm_clustered']['Cluster_Name'].value_counts()
        fig.add_trace(
            go.Pie(labels=cluster_segments.index, values=cluster_segments.values,
                  name="Customer Segments"),
            row=1, col=2
        )
        
        # 3. Product Category Performance
        category_performance = market_insights['product_analysis']['category_performance']
        fig.add_trace(
            go.Bar(x=category_performance.index, y=category_performance['TotalRevenue'],
                  name='Category Revenue', marker_color='lightblue'),
            row=2, col=1
        )
        
        # 4. Seasonal Patterns
        seasonal_data = market_insights['seasonal_analysis']['monthly_patterns']
        months = [calendar.month_name[i] for i in seasonal_data.index]
        fig.add_trace(
            go.Bar(x=months, y=seasonal_data['TotalRevenue'],
                  name='Monthly Revenue', marker_color='lightgreen'),
            row=2, col=2
        )
        
        # 5. Top Products
        top_products = market_insights['product_analysis']['top_products']['top_10_by_revenue'].head(5)
        product_names = [desc[:20] + '...' if len(desc) > 20 else desc 
                        for desc in top_products['Description']]
        fig.add_trace(
            go.Bar(x=product_names, y=top_products['TotalRevenue'],
                  name='Top Products', marker_color='orange'),
            row=3, col=1
        )
        
        # 6. Geographic Distribution (if available)
        if market_insights['geographic_analysis'].get('geographic_data_available', False):
            country_data = market_insights['geographic_analysis']['country_performance'].head(5)
            fig.add_trace(
                go.Pie(labels=country_data.index, values=country_data['TotalRevenue'],
                      name="Geographic Revenue"),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="E-commerce Retail Analytics Dashboard",
            title_x=0.5,
            title_font_size=20,
            showlegend=False
        )
        
        # Save interactive dashboard
        dashboard_path = self.dashboard_path / 'retail_analytics_dashboard.html'
        fig.write_html(str(dashboard_path))
        
        logger.info(f"[OK] Interactive dashboard created: {dashboard_path}")
    
    def create_executive_summary_visualization(self, cleaned_data, segmentation_results, market_insights):
        """
        Create a comprehensive executive summary visualization
        
        Args:
            cleaned_data (pd.DataFrame): Cleaned transaction data
            segmentation_results (dict): Segmentation analysis results
            market_insights (dict): Market analysis results
        """
        logger.info("Creating executive summary visualization...")
        
        # Create a comprehensive summary figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('E-commerce Business Intelligence Executive Summary', fontsize=20, fontweight='bold')
        
        # Key metrics
        total_revenue = cleaned_data['Revenue'].sum()
        total_customers = cleaned_data['CustomerID'].nunique()
        total_orders = cleaned_data['InvoiceNo'].nunique()
        avg_order_value = total_revenue / total_orders
        
        # 1. Key Business Metrics
        metrics = ['Total Revenue', 'Total Customers', 'Total Orders', 'Avg Order Value']
        values = [total_revenue, total_customers, total_orders, avg_order_value]
        colors_metrics = ['gold', 'lightblue', 'lightgreen', 'coral']
        
        bars = axes[0,0].bar(metrics, values, color=colors_metrics)
        axes[0,0].set_title('Key Business Metrics', fontweight='bold')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            if i == 0:  # Revenue
                label = f'£{value:,.0f}'
            elif i == 3:  # AOV
                label = f'£{value:.2f}'
            else:
                label = f'{value:,.0f}'
            axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                          label, ha='center', va='bottom', fontweight='bold')
        
        # 2. Customer Segments
        segments = segmentation_results['clustering_results']['rfm_clustered']['Cluster_Name'].value_counts()
        axes[0,1].pie(segments.values, labels=segments.index, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Customer Segments Distribution', fontweight='bold')
        
        # 3. Monthly Revenue Trend
        monthly_data = market_insights['trend_analysis']['monthly_revenue']
        axes[0,2].plot(monthly_data['YearMonth'].astype(str), monthly_data['MonthlyRevenue'], 
                      marker='o', linewidth=3, markersize=6, color='blue')
        axes[0,2].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Top Product Categories
        category_performance = market_insights['product_analysis']['category_performance'].head(5)
        axes[1,0].bar(category_performance.index, category_performance['TotalRevenue'], 
                     color=self.color_palette[:len(category_performance)])
        axes[1,0].set_title('Top 5 Product Categories', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Seasonal Performance
        seasonal_data = market_insights['seasonal_analysis']['monthly_patterns']
        months = [calendar.month_name[i] for i in seasonal_data.index]
        axes[1,1].bar(months, seasonal_data['SeasonalityIndex'], 
                     color=['red' if x < 100 else 'green' for x in seasonal_data['SeasonalityIndex']])
        axes[1,1].axhline(y=100, color='black', linestyle='--', alpha=0.5)
        axes[1,1].set_title('Monthly Seasonality Index', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Growth Opportunities
        opportunities = market_insights['growth_opportunities']['opportunities']
        opp_counts = {category: len(opps) for category, opps in opportunities.items()}
        
        axes[1,2].bar(opp_counts.keys(), opp_counts.values(), 
                     color=['lightcoral', 'lightyellow', 'lightblue', 'lightgreen'])
        axes[1,2].set_title('Growth Opportunities by Category', fontweight='bold')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'executive_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("[OK] Executive summary visualization created")
    
    def create_all_visualizations(self, cleaned_data, segmentation_results, market_insights):
        """
        Create all visualizations for the analysis
        
        Args:
            cleaned_data (pd.DataFrame): Cleaned transaction data
            segmentation_results (dict): Segmentation analysis results
            market_insights (dict): Market analysis results
        """
        try:
            logger.info("=" * 50)
            logger.info("CREATING ALL VISUALIZATIONS")
            logger.info("=" * 50)
            
            # Create all visualization types
            self.create_customer_segmentation_visualizations(segmentation_results)
            self.create_revenue_trend_visualizations(market_insights)
            self.create_seasonal_analysis_visualizations(market_insights)
            self.create_product_performance_visualizations(market_insights)
            self.create_geographic_visualizations(market_insights)
            self.create_executive_summary_visualization(cleaned_data, segmentation_results, market_insights)
            
            logger.info("=" * 50)
            logger.info("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
            raise
    
    def create_dashboard(self, cleaned_data, segmentation_results, market_insights):
        """
        Create interactive dashboard
        
        Args:
            cleaned_data (pd.DataFrame): Cleaned transaction data
            segmentation_results (dict): Segmentation analysis results
            market_insights (dict): Market analysis results
        """
        try:
            self.create_interactive_dashboard(cleaned_data, segmentation_results, market_insights)
            logger.info("[OK] Interactive dashboard creation completed")
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            raise