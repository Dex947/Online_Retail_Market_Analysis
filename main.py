#!/usr/bin/env python3
"""
E-commerce Market Research & Customer Segmentation Analysis
Main Orchestrator Script

Author: Bryant M.
Date: July 2025
Project: Portfolio Project 2 - Online Retail Analysis
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from data_preprocessing import RetailDataProcessor
from customer_segmentation import CustomerSegmentationAnalyst
from market_analysis import MarketTrendsAnalyzer
from visualization import RetailVisualizationEngine
from report_generator import ReportGenerator

# Create necessary directories first
def create_initial_directories():
    """Create initial project directories"""
    directories = [
        'data/raw', 'data/processed', 'data/external',
        'outputs/reports', 'outputs/visualizations', 'outputs/dashboards',
        'logs', 'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Create directories before logging setup
create_initial_directories()

        # Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetailAnalysisPipeline:
    """
    Main pipeline orchestrator for retail analysis project
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the analysis pipeline"""
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        self.data_processor = RetailDataProcessor(self.config)
        self.segmentation_analyst = CustomerSegmentationAnalyst(self.config)
        self.market_analyzer = MarketTrendsAnalyzer(self.config)
        self.visualization_engine = RetailVisualizationEngine(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        logger.info("Retail Analysis Pipeline initialized successfully")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'data': {
                'raw_data_path': 'data/raw/online_retail.xlsx',
                'processed_data_path': 'data/processed/',
                'date_column': 'InvoiceDate',
                'customer_id_column': 'CustomerID',
                'invoice_column': 'InvoiceNo',
                'product_column': 'Description',
                'quantity_column': 'Quantity',
                'price_column': 'UnitPrice'
            },
            'analysis': {
                'rfm_quantiles': 5,
                'cluster_range': [2, 8],
                'seasonal_analysis': True,
                'country_analysis': True
            },
            'output': {
                'reports_path': 'outputs/reports/',
                'visualizations_path': 'outputs/visualizations/',
                'dashboards_path': 'outputs/dashboards/'
            }
        }
    
    def setup_directories(self):
        """Verify project directories exist"""
        logger.info("Project directories verified")
    
    def run_full_analysis(self):
        """Execute the complete analysis pipeline"""
        try:
            logger.info("="*60)
            logger.info("STARTING E-COMMERCE RETAIL ANALYSIS PIPELINE")
            logger.info("="*60)
            
            # Step 1: Data Preprocessing
            logger.info("Step 1: Data Preprocessing and Cleaning")
            cleaned_data = self.data_processor.process_data()
            logger.info(f"✓ Data processed successfully. Shape: {cleaned_data.shape}")
            
            # Step 2: Customer Segmentation
            logger.info("Step 2: Customer Segmentation Analysis")
            segmentation_results = self.segmentation_analyst.perform_segmentation(cleaned_data)
            logger.info(f"[OK] Customer segmentation completed. Found {segmentation_results['segment_analysis']['segment_summary']['cluster_segments_count']} segments")
            
            # Step 3: Market Trends Analysis
            logger.info("Step 3: Market Trends and Revenue Analysis")
            market_insights = self.market_analyzer.analyze_market_trends(cleaned_data)
            logger.info("[OK] Market trends analysis completed")
            
            # Step 4: Generate Visualizations
            logger.info("Step 4: Creating Visualizations")
            self.visualization_engine.create_all_visualizations(
                cleaned_data, 
                segmentation_results, 
                market_insights
            )
            logger.info("✓ All visualizations created")
            
            # Step 5: Generate Reports
            logger.info("Step 5: Generating Reports")
            self.report_generator.generate_all_reports(
                cleaned_data,
                segmentation_results,
                market_insights
            )
            logger.info("✓ All reports generated")
            
            # Step 6: Create Interactive Dashboard
            logger.info("Step 6: Creating Interactive Dashboard")
            self.visualization_engine.create_dashboard(
                cleaned_data,
                segmentation_results,
                market_insights
            )
            logger.info("✓ Interactive dashboard created")
            
            logger.info("="*60)
            logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _print_summary(self):
        """Print summary of generated outputs"""
        print("\n" + "="*50)
        print("📊 ANALYSIS COMPLETE - OUTPUT SUMMARY")
        print("="*50)
        
        outputs = {
            "📈 Reports": [
                "outputs/reports/executive_summary.html",
                "outputs/reports/technical_report.html",
                "outputs/reports/methodology_documentation.md"
            ],
            "📊 Visualizations": [
                "outputs/visualizations/customer_segments_analysis.png",
                "outputs/visualizations/revenue_trends_monthly.png",
                "outputs/visualizations/seasonal_analysis.png",
                "outputs/visualizations/product_performance.png",
                "outputs/visualizations/geographic_analysis.png"
            ],
            "🎯 Dashboard": [
                "outputs/dashboards/retail_analytics_dashboard.html"
            ],
            "💾 Data": [
                "data/processed/cleaned_retail_data.csv",
                "data/processed/customer_rfm_segments.csv",
                "data/processed/market_insights_summary.csv"
            ]
        }
        
        for category, files in outputs.items():
            print(f"\n{category}:")
            for file in files:
                status = "[OK]" if Path(file).exists() else "[MISSING]"
                print(f"  {status} {file}")
        
        print(f"\n🕒 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)

def main():
    """Main execution function"""
    try:
        # Initialize and run pipeline
        pipeline = RetailAnalysisPipeline()
        pipeline.run_full_analysis()
        
        print("\n[SUCCESS] Your e-commerce analysis is ready for Upwork portfolio!")
        print("\n📋 Next Steps:")
        print("   1. Review generated reports in outputs/reports/")
        print("   2. Check visualizations in outputs/visualizations/")
        print("   3. Open dashboard at outputs/dashboards/retail_analytics_dashboard.html")
        print("   4. Upload to GitHub and link to your Upwork profile")
        
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("Check logs/analysis.log for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()