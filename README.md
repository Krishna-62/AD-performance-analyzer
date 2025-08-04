# AD-performance-analyzer
A full-stack AI-powered dashboard to analyze ad performance, calculate KPIs, predict CTR &amp; conversions, and generate actionable insights with real-time visualizations.
Ad Performance Analyzer
========================

📊 Overview:
------------
The Ad Performance Analyzer is a machine learning-powered tool for analyzing and visualizing digital advertisement data. It enables marketing analysts, campaign managers, and business owners to gain insights into ad effectiveness using metrics like CTR (Click-Through Rate), CPC (Cost Per Click), CPA (Cost Per Acquisition), and Conversion Rates. The tool also builds predictive models to forecast CTR classification and conversion performance.

🎯 Features:
------------
1. **Data Upload & Cleaning**
   - Upload CSV files with ad performance data
   - Auto-cleaning of missing/null values
   - Validates key columns like Impressions, Clicks, Cost, Conversions

2. **KPI Calculation**
   - Computes key marketing KPIs:
     - CTR = Clicks / Impressions
     - CPC = Cost / Clicks
     - CPA = Cost / Conversions
     - ConversionRate = Conversions / Clicks

3. **CTR Classification Model**
   - RandomForestClassifier to predict if CTR is high or not
   - Uses features: Impressions, Clicks, Conversions, Cost
   - GridSearchCV used for hyperparameter tuning
   - Feature importance plots (Bar & Pie)

4. **Conversion Prediction Model**
   - RandomForestRegressor to predict expected conversions
   - Based on Impressions, Clicks, and Cost

5. **Visualization**
   - Bar chart: Top 10 campaigns by predicted conversions
   - Pie chart: Impression share by campaign
   - Actual vs Predicted comparison

6. **Filtering & Export**
   - Date and campaign-level filtering
   - Export filtered data as CSV
   - Download feature importance as image or CSV

🧠 Machine Learning Details:
----------------------------
- **Models**:
  - Classification: RandomForestClassifier (CTR High/Low)
  - Regression: RandomForestRegressor (Conversions)

- **Libraries Used**:
  - pandas, numpy, matplotlib, seaborn
  - sklearn (for modeling)
  - joblib (for saving/loading models)
  - plotly (for rich visualizations)
  - streamlit (for dashboard UI)

📁 Project Structure:
---------------------
AD-performance-analyzer/
├── dashboard/
│ └── app.py # Streamlit dashboard app
├── src/
│ ├── data_loader.py # Load & clean CSV
│ ├── kpi_calculator.py # Calculate marketing KPIs
│ ├── model.py # Train/predict models
│ ├── visualization.py # Visualization utilities
│ └── recommendation_engine.py # (Optional for campaign suggestions)
├── requirements.txt
└── README.txt # This file


✅ How to Run (Development):
----------------------------
1. Make sure Python 3.10+ is installed

2. Create virtual environment (optional but recommended):

3. Install dependencies:

4. Run the dashboard:

💡 Sample CSV Columns:
-----------------------
Your CSV should contain:
- Campaign
- Date
- Impressions
- Clicks
- Conversions
- Cost

📌 Sample Entry:

🎨 Optional Enhancements:
--------------------------
- Firebase integration
- Email report automation
- Real-time database syncing
- Lottie animations for enhanced UI

📬 Contact:
-----------
For suggestions or issues, please raise an issue or contact the developer.

---

