"""
Data Preprocessing Module for E-commerce Retail Analysis

This module handles data loading, cleaning, and preprocessing for the UCI Online Retail dataset.
Includes comprehensive data quality checks and feature engineering.

Author: Bryant M.
Date: July 2025
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RetailDataProcessor:
    """
    Comprehensive data processor for UCI Online Retail dataset
    """
    
    def __init__(self, config):
        """Initialize the data processor with configuration"""
        self.config = config
        self.data_config = config['data']
        self.raw_data_path = self.data_config['raw_data_path']
        self.processed_data_path = self.data_config['processed_data_path']
        
        # Create processed data directory
        Path(self.processed_data_path).mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self):
        """
        Load raw data from Excel file with comprehensive error handling
        
        Returns:
            pd.DataFrame: Raw retail data
        """
        try:
            logger.info(f"Loading data from {self.raw_data_path}")
            
            # Check if file exists
            if not Path(self.raw_data_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.raw_data_path}")
            
            # Load Excel file
            df = pd.read_excel(self.raw_data_path, engine='openpyxl')
            
            logger.info(f"[OK] Raw data loaded successfully. Shape: {df.shape}")
            logger.info(f"[OK] Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            raise
    
    def perform_data_quality_assessment(self, df):
        """
        Comprehensive data quality assessment
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            dict: Data quality report
        """
        logger.info("Performing data quality assessment...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Check for negative quantities and prices
        if 'Quantity' in df.columns:
            quality_report['negative_quantities'] = (df['Quantity'] < 0).sum()
            quality_report['zero_quantities'] = (df['Quantity'] == 0).sum()
        
        if 'UnitPrice' in df.columns:
            quality_report['negative_prices'] = (df['UnitPrice'] < 0).sum()
            quality_report['zero_prices'] = (df['UnitPrice'] == 0).sum()
        
        # Date range analysis
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            quality_report['date_range'] = {
                'start_date': df['InvoiceDate'].min(),
                'end_date': df['InvoiceDate'].max(),
                'date_span_days': (df['InvoiceDate'].max() - df['InvoiceDate'].min()).days
            }
        
        # Customer analysis
        if 'CustomerID' in df.columns:
            quality_report['unique_customers'] = df['CustomerID'].nunique()
            quality_report['missing_customer_ids'] = df['CustomerID'].isnull().sum()
        
        # Product analysis
        if 'StockCode' in df.columns:
            quality_report['unique_products'] = df['StockCode'].nunique()
        
        logger.info("[OK] Data quality assessment completed")
        return quality_report
    
    def clean_data(self, df):
        """
        Comprehensive data cleaning pipeline
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning process...")
        
        initial_rows = len(df)
        
        # 1. Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        logger.info("[OK] Converted InvoiceDate to datetime")
        
        # 2. Remove rows with missing CustomerID (can't analyze customers without ID)
        df = df.dropna(subset=['CustomerID'])
        logger.info(f"[OK] Removed {initial_rows - len(df)} rows with missing CustomerID")
        
        # 3. Remove negative quantities (returns/cancellations for this analysis)
        df = df[df['Quantity'] > 0]
        logger.info(f"[OK] Removed rows with negative/zero quantities. Remaining: {len(df)}")
        
        # 4. Remove zero or negative unit prices
        df = df[df['UnitPrice'] > 0]
        logger.info(f"[OK] Removed rows with zero/negative prices. Remaining: {len(df)}")
        
        # Remove obvious test/invalid entries
        # Remove StockCodes that are clearly not products (like 'POST', 'D', 'M', etc.)
        df = df[~df['StockCode'].str.contains('^[A-Z]{1,}$', regex=True, na=False)]
        
        # 8. Create additional features
        df = self.engineer_features(df)
        
        # 9. Final data validation
        assert len(df) > 0, "No data remaining after cleaning!"
        assert df['CustomerID'].isnull().sum() == 0, "CustomerID nulls still present"
        assert (df['Quantity'] > 0).all(), "Negative quantities still present"
        assert (df['UnitPrice'] > 0).all(), "Non-positive prices still present"
        
        logger.info(f"[OK] Data cleaning completed. Final shape: {df.shape}")
        logger.info(f"[OK] Data reduction: {((initial_rows - len(df)) / initial_rows * 100):.1f}%")
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features for analysis
        
        Args:
            df (pd.DataFrame): Cleaned dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Engineering additional features...")
        
        # 1. Revenue calculation
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
        # 2. Date-based features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Weekday'] = df['InvoiceDate'].dt.dayofweek
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
        
        # 3. Invoice-level aggregations
        invoice_stats = df.groupby('InvoiceNo').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'StockCode': 'nunique'
        }).rename(columns={
            'Revenue': 'InvoiceValue',
            'Quantity': 'InvoiceQuantity',
            'StockCode': 'UniqueProductsPerInvoice'
        })
        
        df = df.merge(invoice_stats, on='InvoiceNo', how='left')
        
        # 4. Product category inference (simplified)
        df['ProductCategory'] = self.categorize_products(df['Description'])
        
        # 5. Customer frequency indicators (preliminary)
        customer_invoice_count = df.groupby('CustomerID')['InvoiceNo'].nunique()
        df['CustomerInvoiceCount'] = df['CustomerID'].map(customer_invoice_count)
        
        logger.info("[OK] Feature engineering completed")
        logger.info(f"[OK] Added features: Revenue, Date components, Invoice stats, Product categories")
        
        return df
    
    def categorize_products(self, descriptions):
        """
        Simple product categorization based on keywords
        
        Args:
            descriptions (pd.Series): Product descriptions
            
        Returns:
            pd.Series: Product categories
        """
        categories = pd.Series(['OTHER'] * len(descriptions), index=descriptions.index)
        
        # Define category keywords
        category_keywords = {
            'HOME_DECOR': ['DECORATION', 'DECOR', 'ORNAMENT', 'FRAME', 'CANDLE', 'LIGHT'],
            'KITCHEN': ['KITCHEN', 'CUP', 'MUG', 'PLATE', 'BOWL', 'SPOON', 'FORK'],
            'GARDEN': ['GARDEN', 'PLANT', 'FLOWER', 'SEED', 'WATERING'],
            'STATIONERY': ['PENCIL', 'PEN', 'PAPER', 'NOTEBOOK', 'CARD', 'ENVELOPE'],
            'TOYS': ['TOY', 'GAME', 'PLAY', 'CHILDREN', 'KIDS'],
            'CLOTHING': ['CLOTHING', 'SHIRT', 'DRESS', 'HAT', 'SOCK', 'SHOE'],
            'CHRISTMAS': ['CHRISTMAS', 'XMAS', 'SANTA', 'SNOWMAN', 'TREE']
        }
        
        for category, keywords in category_keywords.items():
            mask = descriptions.str.contains('|'.join(keywords), na=False, regex=True)
            categories[mask] = category
        
        return categories
    
    def save_processed_data(self, df, quality_report):
        """
        Save processed data and quality report
        
        Args:
            df (pd.DataFrame): Processed dataframe
            quality_report (dict): Data quality assessment results
        """
        # Save main processed dataset
        output_path = Path(self.processed_data_path) / 'cleaned_retail_data.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"[OK] Processed data saved to {output_path}")
        
        # Save data quality report
        quality_path = Path(self.processed_data_path) / 'data_quality_report.txt'
        with open(quality_path, 'w') as f:
            f.write("E-COMMERCE DATA QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in quality_report.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        logger.info(f"[OK] Quality report saved to {quality_path}")
        
        # Save summary statistics
        summary_path = Path(self.processed_data_path) / 'data_summary_stats.csv'
        summary_stats = df.describe(include='all')
        summary_stats.to_csv(summary_path)
        logger.info(f"[OK] Summary statistics saved to {summary_path}")
    
    def process_data(self):
        """
        Main data processing pipeline
        
        Returns:
            pd.DataFrame: Fully processed and cleaned data
        """
        try:
            # Load raw data
            raw_data = self.load_raw_data()
            
            # Assess data quality
            quality_report = self.perform_data_quality_assessment(raw_data)
            
            # Clean and process data
            cleaned_data = self.clean_data(raw_data)
            
            # Save processed data
            self.save_processed_data(cleaned_data, quality_report)
            
            # Final summary
            logger.info("=" * 50)
            logger.info("DATA PROCESSING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Original rows: {quality_report['total_rows']:,}")
            logger.info(f"Processed rows: {len(cleaned_data):,}")
            logger.info(f"Data retention: {(len(cleaned_data) / quality_report['total_rows'] * 100):.1f}%")
            logger.info(f"Unique customers: {cleaned_data['CustomerID'].nunique():,}")
            logger.info(f"Unique products: {cleaned_data['StockCode'].nunique():,}")
            logger.info(f"Date range: {cleaned_data['InvoiceDate'].min()} to {cleaned_data['InvoiceDate'].max()}")
            logger.info(f"Total revenue: £{cleaned_data['Revenue'].sum():,.2f}")
            logger.info("=" * 50)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

# Utility functions for data validation
def validate_processed_data(df):
    """
    Validate that processed data meets quality standards
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        bool: True if data passes validation
    """
    validations = {
        'No missing CustomerID': df['CustomerID'].isnull().sum() == 0,
        'Positive quantities': (df['Quantity'] > 0).all(),
        'Positive prices': (df['UnitPrice'] > 0).all(),
        'Valid dates': pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']),
        'Revenue calculated': 'Revenue' in df.columns,
        'Minimum records': len(df) > 1000,
        'Multiple customers': df['CustomerID'].nunique() > 100,
        'Multiple products': df['StockCode'].nunique() > 100
    }
    
    all_passed = True
    for validation, passed in validations.items():
        status = "[PASS]" if passed else "[FAIL]"
        logger.info(f"{status}: {validation}")
        if not passed:
            all_passed = False
    
    def engineer_features(self, df):
        """
        Create additional features for analysis
        
        Args:
            df (pd.DataFrame): Cleaned dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Engineering additional features...")
        
        # 1. Revenue calculation
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        
        # 2. Date-based features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['Weekday'] = df['InvoiceDate'].dt.dayofweek
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
        
        # 3. Invoice-level aggregations
        invoice_stats = df.groupby('InvoiceNo').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'StockCode': 'nunique'
        }).rename(columns={
            'Revenue': 'InvoiceValue',
            'Quantity': 'InvoiceQuantity',
            'StockCode': 'UniqueProductsPerInvoice'
        })
        
        df = df.merge(invoice_stats, on='InvoiceNo', how='left')
        
        # 4. Product category inference (simplified)
        df['ProductCategory'] = self.categorize_products(df['Description'])
        
        # 5. Customer frequency indicators (preliminary)
        customer_invoice_count = df.groupby('CustomerID')['InvoiceNo'].nunique()
        df['CustomerInvoiceCount'] = df['CustomerID'].map(customer_invoice_count)
        
        logger.info("✓ Feature engineering completed")
        logger.info(f"✓ Added features: Revenue, Date components, Invoice stats, Product categories")
        
        return df
    
    def categorize_products(self, descriptions):
        """
        Simple product categorization based on keywords
        
        Args:
            descriptions (pd.Series): Product descriptions
            
        Returns:
            pd.Series: Product categories
        """
        categories = pd.Series(['OTHER'] * len(descriptions), index=descriptions.index)
        
        # Define category keywords
        category_keywords = {
            'HOME_DECOR': ['DECORATION', 'DECOR', 'ORNAMENT', 'FRAME', 'CANDLE', 'LIGHT'],
            'KITCHEN': ['KITCHEN', 'CUP', 'MUG', 'PLATE', 'BOWL', 'SPOON', 'FORK'],
            'GARDEN': ['GARDEN', 'PLANT', 'FLOWER', 'SEED', 'WATERING'],
            'STATIONERY': ['PENCIL', 'PEN', 'PAPER', 'NOTEBOOK', 'CARD', 'ENVELOPE'],
            'TOYS': ['TOY', 'GAME', 'PLAY', 'CHILDREN', 'KIDS'],
            'CLOTHING': ['CLOTHING', 'SHIRT', 'DRESS', 'HAT', 'SOCK', 'SHOE'],
            'CHRISTMAS': ['CHRISTMAS', 'XMAS', 'SANTA', 'SNOWMAN', 'TREE']
        }
        
        for category, keywords in category_keywords.items():
            mask = descriptions.str.contains('|'.join(keywords), na=False, regex=True)
            categories[mask] = category
        
        return categories
    
    def save_processed_data(self, df, quality_report):
        """
        Save processed data and quality report
        
        Args:
            df (pd.DataFrame): Processed dataframe
            quality_report (dict): Data quality assessment results
        """
        # Save main processed dataset
        output_path = Path(self.processed_data_path) / 'cleaned_retail_data.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")
        
        # Save data quality report
        quality_path = Path(self.processed_data_path) / 'data_quality_report.txt'
        with open(quality_path, 'w') as f:
            f.write("E-COMMERCE DATA QUALITY ASSESSMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in quality_report.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        
        logger.info(f"✓ Quality report saved to {quality_path}")
        
        # Save summary statistics
        summary_path = Path(self.processed_data_path) / 'data_summary_stats.csv'
        summary_stats = df.describe(include='all')
        summary_stats.to_csv(summary_path)
        logger.info(f"✓ Summary statistics saved to {summary_path}")
    
    def process_data(self):
        """
        Main data processing pipeline
        
        Returns:
            pd.DataFrame: Fully processed and cleaned data
        """
        try:
            # Load raw data
            raw_data = self.load_raw_data()
            
            # Assess data quality
            quality_report = self.perform_data_quality_assessment(raw_data)
            
            # Clean and process data
            cleaned_data = self.clean_data(raw_data)
            
            # Validate processed data
            if not validate_processed_data(cleaned_data):
                logger.error("Processed data failed validation")
                raise ValueError("Processed data failed validation")
            
            # Save processed data
            self.save_processed_data(cleaned_data, quality_report)
            
            # Final summary
            logger.info("=" * 50)
            logger.info("DATA PROCESSING SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Original rows: {quality_report['total_rows']:,}")
            logger.info(f"Processed rows: {len(cleaned_data):,}")
            logger.info(f"Data retention: {(len(cleaned_data) / quality_report['total_rows'] * 100):.1f}%")
            logger.info(f"Unique customers: {cleaned_data['CustomerID'].nunique():,}")
            logger.info(f"Unique products: {cleaned_data['StockCode'].nunique():,}")
            logger.info(f"Date range: {cleaned_data['InvoiceDate'].min()} to {cleaned_data['InvoiceDate'].max()}")
            logger.info(f"Total revenue: £{cleaned_data['Revenue'].sum():,.2f}")
            logger.info("=" * 50)
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise

# Utility functions for data validation
def validate_processed_data(df):
    """
    Validate that processed data meets quality standards
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        bool: True if data passes validation
    """
    validations = {
        'No missing CustomerID': df['CustomerID'].isnull().sum() == 0,
        'Positive quantities': (df['Quantity'] > 0).all(),
        'Positive prices': (df['UnitPrice'] > 0).all(),
        'Valid dates': pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']),
        'Revenue calculated': 'Revenue' in df.columns,
        'Minimum records': len(df) > 1000,
        'Multiple customers': df['CustomerID'].nunique() > 100,
        'Multiple products': df['StockCode'].nunique() > 100
    }
    
    all_passed = True
    for validation, passed in validations.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {validation}")
        if not passed:
            all_passed = False
    
    return all_passed