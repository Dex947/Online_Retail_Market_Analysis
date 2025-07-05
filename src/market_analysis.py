"""
Market Trends Analysis Module

Analyzes market trends, seasonal patterns, product performance, and revenue optimization
opportunities for e-commerce retail data.

Author: Bryant M.
Date: July 2025
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import pearsonr
import calendar

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MarketTrendsAnalyzer:
    """
    Comprehensive market trends and revenue analysis
    """
    
    def __init__(self, config):
        """Initialize market trends analyzer"""
        self.config = config
        self.analysis_config = config['analysis']
        self.output_path = config['output']['reports_path']
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
    def analyze_revenue_trends(self, df):
        """
        Analyze revenue trends over time
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Revenue trend analysis results
        """
        logger.info("Analyzing revenue trends...")
        
        # Daily revenue trends
        daily_revenue = df.groupby(df['InvoiceDate'].dt.date).agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique',
            'Quantity': 'sum'
        }).reset_index()
        daily_revenue.columns = ['Date', 'DailyRevenue', 'DailyOrders', 'DailyCustomers', 'DailyQuantity']
        daily_revenue['AvgOrderValue'] = daily_revenue['DailyRevenue'] / daily_revenue['DailyOrders']
        
        # Weekly revenue trends
        df['WeekStart'] = df['InvoiceDate'].dt.to_period('W').dt.start_time
        weekly_revenue = df.groupby('WeekStart').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique',
            'Quantity': 'sum'
        }).reset_index()
        weekly_revenue.columns = ['WeekStart', 'WeeklyRevenue', 'WeeklyOrders', 'WeeklyCustomers', 'WeeklyQuantity']
        
        # Monthly revenue trends
        monthly_revenue = df.groupby('YearMonth').agg({
            'Revenue': 'sum',
            'InvoiceNo': 'nunique',
            'CustomerID': 'nunique',
            'Quantity': 'sum'
        }).reset_index()
        monthly_revenue.columns = ['YearMonth', 'MonthlyRevenue', 'MonthlyOrders', 'MonthlyCustomers', 'MonthlyQuantity']
        monthly_revenue['MonthlyAvgOrderValue'] = monthly_revenue['MonthlyRevenue'] / monthly_revenue['MonthlyOrders']
        
        # Calculate growth rates
        monthly_revenue['RevenueGrowthRate'] = monthly_revenue['MonthlyRevenue'].pct_change() * 100
        monthly_revenue['CustomerGrowthRate'] = monthly_revenue['MonthlyCustomers'].pct_change() * 100
        monthly_revenue['OrderGrowthRate'] = monthly_revenue['MonthlyOrders'].pct_change() * 100
        
        # Revenue trend statistics
        total_revenue = df['Revenue'].sum()
        avg_daily_revenue = daily_revenue['DailyRevenue'].mean()
        revenue_volatility = daily_revenue['DailyRevenue'].std()
        
        # Correlation analysis between time and revenue
        daily_revenue['DayNumber'] = range(len(daily_revenue))
        revenue_time_correlation, p_value = pearsonr(daily_revenue['DayNumber'], daily_revenue['DailyRevenue'])
        
        trend_analysis = {
            'daily_revenue': daily_revenue,
            'weekly_revenue': weekly_revenue,
            'monthly_revenue': monthly_revenue,
            'summary_stats': {
                'total_revenue': total_revenue,
                'avg_daily_revenue': avg_daily_revenue,
                'revenue_volatility': revenue_volatility,
                'revenue_time_correlation': revenue_time_correlation,
                'trend_significance': p_value < 0.05,
                'peak_revenue_day': daily_revenue.loc[daily_revenue['DailyRevenue'].idxmax(), 'Date'],
                'peak_revenue_amount': daily_revenue['DailyRevenue'].max(),
                'avg_monthly_growth': monthly_revenue['RevenueGrowthRate'].mean()
            }
        }
        
        logger.info("[OK] Revenue trends analysis completed")
        logger.info(f"[OK] Total revenue analyzed: £{total_revenue:,.2f}")
        logger.info(f"[OK] Average daily revenue: £{avg_daily_revenue:,.2f}")
        logger.info(f"[OK] Revenue-time correlation: {revenue_time_correlation:.3f}")
        
        return trend_analysis
    
    def analyze_seasonal_patterns(self, df):
        """
        Analyze seasonal patterns in sales and revenue
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Seasonal analysis results
        """
        logger.info("Analyzing seasonal patterns...")
        
        # Monthly seasonality
        monthly_seasonality = df.groupby('Month').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'CustomerID': 'nunique'
        }).round(2)
        monthly_seasonality.columns = ['TotalRevenue', 'AvgRevenue', 'TransactionCount', 'TotalQuantity', 'UniqueCustomers']
        
        # Ensure we have all 12 months (fill missing months with 0)
        monthly_seasonality = monthly_seasonality.reindex(range(1, 13), fill_value=0)
        monthly_seasonality['Month_Name'] = monthly_seasonality.index.map(lambda x: calendar.month_name[x])
        
        # Quarterly seasonality
        quarterly_seasonality = df.groupby('Quarter').agg({
            'Revenue': ['sum', 'mean'],
            'Quantity': 'sum',
            'CustomerID': 'nunique'
        }).round(2)
        quarterly_seasonality.columns = ['TotalRevenue', 'AvgRevenue', 'TotalQuantity', 'UniqueCustomers']
        
        # Ensure we have all 4 quarters
        quarterly_seasonality = quarterly_seasonality.reindex(range(1, 5), fill_value=0)
        
        # Day of week patterns
        dow_patterns = df.groupby('Weekday').agg({
            'Revenue': ['sum', 'mean'],
            'Quantity': 'sum',
            'CustomerID': 'nunique'
        }).round(2)
        dow_patterns.columns = ['TotalRevenue', 'AvgRevenue', 'TotalQuantity', 'UniqueCustomers']
        
        # Ensure we have all 7 days (0=Monday to 6=Sunday)
        full_dow_patterns = dow_patterns.reindex(range(7), fill_value=0)
        full_dow_patterns['DayName'] = full_dow_patterns.index.map(
            {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        )
        
        # Holiday/special period analysis (focusing on Christmas period)
        df['IsDecember'] = df['Month'] == 12
        df['IsNovDec'] = df['Month'].isin([11, 12])
        
        holiday_analysis = {
            'december_revenue': df[df['IsDecember']]['Revenue'].sum(),
            'december_share': df[df['IsDecember']]['Revenue'].sum() / df['Revenue'].sum() * 100,
            'nov_dec_revenue': df[df['IsNovDec']]['Revenue'].sum(),
            'nov_dec_share': df[df['IsNovDec']]['Revenue'].sum() / df['Revenue'].sum() * 100,
            'december_avg_order': df[df['IsDecember']]['Revenue'].mean(),
            'non_december_avg_order': df[~df['IsDecember']]['Revenue'].mean()
        }
        
        # Calculate seasonality indices (monthly revenue / average monthly revenue)
        avg_monthly_revenue = monthly_seasonality['TotalRevenue'].mean()
        if avg_monthly_revenue > 0:
            monthly_seasonality['SeasonalityIndex'] = (monthly_seasonality['TotalRevenue'] / avg_monthly_revenue * 100).round(2)
        else:
            monthly_seasonality['SeasonalityIndex'] = 100.0  # Default to 100 if no revenue
        
        # Statistical test for seasonal differences
        monthly_groups = [df[df['Month'] == month]['Revenue'] for month in range(1, 13)]
        f_stat, p_value = stats.f_oneway(*monthly_groups)
        
        seasonal_analysis = {
            'monthly_patterns': monthly_seasonality,
            'quarterly_patterns': quarterly_seasonality,
            'day_of_week_patterns': full_dow_patterns,
            'holiday_analysis': holiday_analysis,
            'seasonality_stats': {
                'peak_month': monthly_seasonality['TotalRevenue'].idxmax() if monthly_seasonality['TotalRevenue'].max() > 0 else 1,
                'peak_month_name': calendar.month_name[monthly_seasonality['TotalRevenue'].idxmax() if monthly_seasonality['TotalRevenue'].max() > 0 else 1],
                'lowest_month': monthly_seasonality['TotalRevenue'].idxmin() if monthly_seasonality['TotalRevenue'].min() >= 0 else 1,
                'lowest_month_name': calendar.month_name[monthly_seasonality['TotalRevenue'].idxmin() if monthly_seasonality['TotalRevenue'].min() >= 0 else 1],
                'peak_quarter': quarterly_seasonality['TotalRevenue'].idxmax() if quarterly_seasonality['TotalRevenue'].max() > 0 else 1,
                'best_weekday': full_dow_patterns['TotalRevenue'].idxmax() if full_dow_patterns['TotalRevenue'].max() > 0 else 0,
                'seasonality_anova_f': f_stat,
                'seasonality_anova_p': p_value,
                'significant_seasonality': p_value < 0.05
            }
        }
        
        logger.info("[OK] Seasonal patterns analysis completed")
        logger.info(f"[OK] Peak month: {seasonal_analysis['seasonality_stats']['peak_month_name']}")
        logger.info(f"[OK] Significant seasonality: {seasonal_analysis['seasonality_stats']['significant_seasonality']}")
        
        return seasonal_analysis
    
    def analyze_product_performance(self, df):
        """
        Analyze product performance and categories
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Product performance analysis
        """
        logger.info("Analyzing product performance...")
        
        # Product-level analysis
        product_performance = df.groupby('StockCode').agg({
            'Description': 'first',
            'Revenue': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'UnitPrice': ['mean', 'min', 'max'],
            'CustomerID': 'nunique'
        }).round(2)
        
        # Flatten column names
        product_performance.columns = [
            'Description', 'TotalRevenue', 'AvgRevenue', 'OrderCount', 
            'TotalQuantity', 'AvgPrice', 'MinPrice', 'MaxPrice', 'UniqueCustomers'
        ]
        
        # Calculate additional metrics
        product_performance['RevenuePerCustomer'] = product_performance['TotalRevenue'] / product_performance['UniqueCustomers']
        product_performance['RevenueShare'] = (product_performance['TotalRevenue'] / df['Revenue'].sum() * 100).round(3)
        
        # Sort by total revenue
        product_performance = product_performance.sort_values('TotalRevenue', ascending=False)
        
        # Top products analysis
        top_products = {
            'top_10_by_revenue': product_performance.head(10),
            'top_10_by_quantity': product_performance.sort_values('TotalQuantity', ascending=False).head(10),
            'top_10_by_customers': product_performance.sort_values('UniqueCustomers', ascending=False).head(10)
        }
        
        # Category-level analysis
        category_performance = df.groupby('ProductCategory').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'CustomerID': 'nunique',
            'StockCode': 'nunique'
        }).round(2)
        
        category_performance.columns = [
            'TotalRevenue', 'AvgRevenue', 'OrderCount', 
            'TotalQuantity', 'UniqueCustomers', 'UniqueProducts'
        ]
        category_performance['RevenueShare'] = (category_performance['TotalRevenue'] / df['Revenue'].sum() * 100).round(2)
        category_performance = category_performance.sort_values('TotalRevenue', ascending=False)
        
        # Product concentration analysis (Pareto principle)
        total_revenue = df['Revenue'].sum()
        cumulative_revenue = product_performance['TotalRevenue'].cumsum()
        cumulative_percentage = (cumulative_revenue / total_revenue * 100)
        
        # Find products that make up 80% of revenue
        products_80_percent = len(cumulative_percentage[cumulative_percentage <= 80])
        total_products = len(product_performance)
        pareto_ratio = products_80_percent / total_products * 100
        
        # ABC Analysis
        product_performance['CumulativeRevenueShare'] = cumulative_percentage
        product_performance['ABC_Category'] = pd.cut(
            product_performance['CumulativeRevenueShare'],
            bins=[0, 80, 95, 100],
            labels=['A', 'B', 'C']
        )
        
        abc_analysis = product_performance['ABC_Category'].value_counts()
        
        product_analysis = {
            'product_performance': product_performance,
            'category_performance': category_performance,
            'top_products': top_products,
            'concentration_analysis': {
                'products_80_percent_revenue': products_80_percent,
                'total_products': total_products,
                'pareto_ratio': pareto_ratio,
                'abc_analysis': abc_analysis
            },
            'performance_stats': {
                'total_unique_products': df['StockCode'].nunique(),
                'avg_revenue_per_product': product_performance['TotalRevenue'].mean(),
                'top_product_revenue': product_performance['TotalRevenue'].iloc[0],
                'top_product_name': product_performance['Description'].iloc[0],
                'top_category': category_performance.index[0],
                'top_category_revenue_share': category_performance['RevenueShare'].iloc[0]
            }
        }
        
        logger.info("[OK] Product performance analysis completed")
        logger.info(f"[OK] Total unique products: {product_analysis['performance_stats']['total_unique_products']:,}")
        logger.info(f"[OK] Top product: {product_analysis['performance_stats']['top_product_name']}")
        logger.info(f"[OK] Products for 80% revenue: {products_80_percent} ({pareto_ratio:.1f}%)")
        
        return product_analysis
    
    def analyze_geographic_patterns(self, df):
        """
        Analyze geographic sales patterns by country
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Geographic analysis results
        """
        logger.info("Analyzing geographic patterns...")
        
        if 'Country' not in df.columns:
            logger.warning("Country column not found. Skipping geographic analysis.")
            return {'geographic_data_available': False}
        
        # Country-level analysis
        country_performance = df.groupby('Country').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Quantity': 'sum',
            'CustomerID': 'nunique',
            'UnitPrice': 'mean'
        }).round(2)
        
        country_performance.columns = [
            'TotalRevenue', 'AvgRevenue', 'OrderCount', 
            'TotalQuantity', 'UniqueCustomers', 'AvgUnitPrice'
        ]
        
        # Calculate additional metrics
        total_revenue = df['Revenue'].sum()
        country_performance['RevenueShare'] = (country_performance['TotalRevenue'] / total_revenue * 100).round(2)
        country_performance['AvgRevenuePerCustomer'] = country_performance['TotalRevenue'] / country_performance['UniqueCustomers']
        country_performance = country_performance.sort_values('TotalRevenue', ascending=False)
        
        # International vs domestic analysis (assuming UK is domestic)
        uk_revenue = country_performance.loc['United Kingdom', 'TotalRevenue'] if 'United Kingdom' in country_performance.index else 0
        international_revenue = total_revenue - uk_revenue
        
        geographic_analysis = {
            'country_performance': country_performance,
            'geographic_stats': {
                'total_countries': len(country_performance),
                'top_country': country_performance.index[0],
                'top_country_revenue_share': country_performance['RevenueShare'].iloc[0],
                'uk_revenue_share': (uk_revenue / total_revenue * 100) if uk_revenue > 0 else 0,
                'international_revenue_share': (international_revenue / total_revenue * 100),
                'avg_revenue_per_country': country_performance['TotalRevenue'].mean()
            },
            'geographic_data_available': True
        }
        
        logger.info("[OK] Geographic patterns analysis completed")
        logger.info(f"[OK] Total countries: {geographic_analysis['geographic_stats']['total_countries']}")
        logger.info(f"[OK] Top country: {geographic_analysis['geographic_stats']['top_country']}")
        
        return geographic_analysis
    
    def identify_growth_opportunities(self, df, seasonal_analysis, product_analysis):
        """
        Identify revenue growth opportunities based on analysis
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            seasonal_analysis (dict): Seasonal analysis results
            product_analysis (dict): Product analysis results
            
        Returns:
            dict: Growth opportunities and recommendations
        """
        logger.info("Identifying growth opportunities...")
        
        opportunities = {
            'seasonal_opportunities': [],
            'product_opportunities': [],
            'customer_opportunities': [],
            'operational_opportunities': []
        }
        
        # Seasonal opportunities
        monthly_patterns = seasonal_analysis['monthly_patterns']
        low_months = monthly_patterns[monthly_patterns['SeasonalityIndex'] < 90]
        
        for month in low_months.index:
            month_name = calendar.month_name[month]
            seasonal_index = low_months.loc[month, 'SeasonalityIndex']
            opportunities['seasonal_opportunities'].append({
                'type': 'Seasonal Boost',
                'description': f'Target {month_name} for promotional campaigns',
                'current_performance': f'{seasonal_index:.1f}% of average',
                'potential_impact': 'Medium',
                'recommendation': f'Develop targeted marketing campaigns for {month_name} to boost sales'
            })
        
        # Product opportunities
        category_performance = product_analysis['category_performance']
        underperforming_categories = category_performance[category_performance['RevenueShare'] < 5]
        
        for category in underperforming_categories.index:
            revenue_share = underperforming_categories.loc[category, 'RevenueShare']
            opportunities['product_opportunities'].append({
                'type': 'Product Category Growth',
                'description': f'Expand {category} category',
                'current_performance': f'{revenue_share:.1f}% revenue share',
                'potential_impact': 'High',
                'recommendation': f'Increase product variety and marketing for {category} category'
            })
        
        # Customer opportunities
        # This would be enhanced with segmentation data
        avg_customer_value = df.groupby('CustomerID')['Revenue'].sum().mean()
        opportunities['customer_opportunities'].append({
            'type': 'Customer Value Increase',
            'description': 'Increase average customer lifetime value',
            'current_performance': f'£{avg_customer_value:.2f} avg customer value',
            'potential_impact': 'High',
            'recommendation': 'Implement loyalty programs and cross-selling strategies'
        })
        
        # Operational opportunities
        # Based on day of week patterns
        dow_patterns_for_analysis = seasonal_analysis['day_of_week_patterns']
        best_day_revenue = dow_patterns_for_analysis['TotalRevenue'].max()
        worst_day_revenue = dow_patterns_for_analysis['TotalRevenue'].min()
        
        if worst_day_revenue > 0:  # Avoid division by zero
            improvement_potential = (best_day_revenue - worst_day_revenue) / worst_day_revenue * 100
        else:
            improvement_potential = 0
        
        if improvement_potential > 20:
            opportunities['operational_opportunities'].append({
                'type': 'Day-of-Week Optimization',
                'description': 'Balance daily sales distribution',
                'current_performance': f'{improvement_potential:.1f}% variance between best/worst days',
                'potential_impact': 'Medium',
                'recommendation': 'Implement day-specific promotions to balance weekly sales'
            })
        
        # Revenue growth projections
        current_monthly_avg = df['Revenue'].sum() / df['YearMonth'].nunique()
        
        growth_projections = {
            'conservative_5_percent': current_monthly_avg * 1.05 * 12,
            'moderate_15_percent': current_monthly_avg * 1.15 * 12,
            'aggressive_25_percent': current_monthly_avg * 1.25 * 12
        }
        
        growth_opportunities = {
            'opportunities': opportunities,
            'growth_projections': growth_projections,
            'current_annual_projection': current_monthly_avg * 12,
            'priority_recommendations': self._prioritize_recommendations(opportunities)
        }
        
        logger.info("[OK] Growth opportunities analysis completed")
        logger.info(f"[OK] Total opportunities identified: {sum(len(opps) for opps in opportunities.values())}")
        
        return growth_opportunities
    
    def _prioritize_recommendations(self, opportunities):
        """
        Prioritize recommendations based on impact and effort
        
        Args:
            opportunities (dict): All identified opportunities
            
        Returns:
            list: Prioritized recommendations
        """
        all_opportunities = []
        for category, opps in opportunities.items():
            for opp in opps:
                opp['category'] = category
                all_opportunities.append(opp)
        
        # Simple prioritization based on potential impact
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        
        prioritized = sorted(
            all_opportunities,
            key=lambda x: priority_map.get(x.get('potential_impact', 'Low'), 1),
            reverse=True
        )
        
        return prioritized[:5]  # Top 5 recommendations
    
    def save_market_analysis_results(self, trend_analysis, seasonal_analysis, product_analysis, 
                                   geographic_analysis, growth_opportunities):
        """
        Save all market analysis results to files
        
        Args:
            Various analysis result dictionaries
        """
        output_dir = Path(self.config['data']['processed_data_path'])
        
        # Save trend analysis
        trend_analysis['monthly_revenue'].to_csv(output_dir / 'monthly_revenue_trends.csv', index=False)
        trend_analysis['daily_revenue'].to_csv(output_dir / 'daily_revenue_trends.csv', index=False)
        
        # Save seasonal analysis
        seasonal_analysis['monthly_patterns'].to_csv(output_dir / 'monthly_seasonal_patterns.csv')
        seasonal_analysis['quarterly_patterns'].to_csv(output_dir / 'quarterly_patterns.csv')
        
        # Save product analysis
        product_analysis['product_performance'].to_csv(output_dir / 'product_performance_analysis.csv')
        product_analysis['category_performance'].to_csv(output_dir / 'category_performance_analysis.csv')
        
        # Save geographic analysis if available
        if geographic_analysis.get('geographic_data_available', False):
            geographic_analysis['country_performance'].to_csv(output_dir / 'country_performance_analysis.csv')
        
        # Save comprehensive market insights summary
        summary_path = output_dir / 'market_insights_summary.csv'
        market_summary = pd.DataFrame([{
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
            'Total_Revenue': trend_analysis['summary_stats']['total_revenue'],
            'Avg_Daily_Revenue': trend_analysis['summary_stats']['avg_daily_revenue'],
            'Peak_Month': seasonal_analysis['seasonality_stats']['peak_month_name'],
            'Top_Product_Category': product_analysis['performance_stats']['top_category'],
            'Unique_Products': product_analysis['performance_stats']['total_unique_products'],
            'Growth_Opportunities_Count': sum(len(opps) for opps in growth_opportunities['opportunities'].values()),
            'Conservative_Growth_Projection': growth_opportunities['growth_projections']['conservative_5_percent'],
            'Moderate_Growth_Projection': growth_opportunities['growth_projections']['moderate_15_percent']
        }])
        market_summary.to_csv(summary_path, index=False)
        
        logger.info(f"[OK] Market analysis results saved to {output_dir}")
    
    def analyze_market_trends(self, df):
        """
        Main market trends analysis pipeline
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            
        Returns:
            dict: Complete market analysis results
        """
        try:
            logger.info("=" * 50)
            logger.info("STARTING MARKET TRENDS ANALYSIS")
            logger.info("=" * 50)
            
            # Step 1: Revenue trends analysis
            trend_analysis = self.analyze_revenue_trends(df)
            
            # Step 2: Seasonal patterns analysis
            seasonal_analysis = self.analyze_seasonal_patterns(df)
            
            # Step 3: Product performance analysis
            product_analysis = self.analyze_product_performance(df)
            
            # Step 4: Geographic patterns analysis
            geographic_analysis = self.analyze_geographic_patterns(df)
            
            # Step 5: Growth opportunities identification
            growth_opportunities = self.identify_growth_opportunities(
                df, seasonal_analysis, product_analysis
            )
            
            # Step 6: Save all results
            self.save_market_analysis_results(
                trend_analysis, seasonal_analysis, product_analysis,
                geographic_analysis, growth_opportunities
            )
            
            # Compile final results
            market_insights = {
                'trend_analysis': trend_analysis,
                'seasonal_analysis': seasonal_analysis,
                'product_analysis': product_analysis,
                'geographic_analysis': geographic_analysis,
                'growth_opportunities': growth_opportunities
            }
            
            logger.info("=" * 50)
            logger.info("MARKET TRENDS ANALYSIS COMPLETED SUCCESSFULLY")
            logger.info("=" * 50)
            
            return market_insights
            
        except Exception as e:
            logger.error(f"Market trends analysis failed: {str(e)}")
            raise