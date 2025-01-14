# Rossmann Pharmaceuticals Sales Forecasting

## Overview
This project aims to build an end-to-end machine learning pipeline to forecast sales in Rossmann Pharmaceuticals stores across various cities. The predictions will help the finance team plan six weeks ahead of time. The solution incorporates advanced data analysis, feature engineering, machine learning, and deep learning techniques.

## Table of Contents
1. [Business Need](#business-need)
2. [Data and Features](#data-and-features)
3. [Learning Outcomes](#learning-outcomes)
4. [Project Objectives](#project-objectives)
5. [Setup Instructions](#setup-instructions)
6. [Task Breakdown](#task-breakdown)
    - [Exploration of Customer Purchasing Behavior](#task-1---exploration-of-customer-purchasing-behavior)
    - [Prediction of Store Sales](#task-2---prediction-of-store-sales)
    - [Deep Learning Approach](#task-2.6-building-model-with-deep-learning)
    - [Model Serving API Call](#task-3---model-serving-api-call)
7. [File Structure](#file-structure)
8. [How to Use](#how-to-use)

## Business Need
Rossmann Pharmaceuticals requires a sales forecasting solution to aid the finance team in planning six weeks in advance. Managers currently rely on personal judgment for sales forecasting, which can be improved using data-driven machine learning models.

Key features identified for prediction include:
- Promotions
- Competition details
- School and state holidays
- Seasonality
- Locality

The solution must deliver actionable insights and accurate predictions.

## Data and Features
The dataset includes the following fields:
- **Id**: Represents a (Store, Date) duple in the test set.
- **Store**: A unique ID for each store.
- **Sales**: Daily turnover (target variable).
- **Customers**: Number of customers per day.
- **Open**: Indicator for store operation (1: Open, 0: Closed).
- **StateHoliday**: Type of state holiday.
- **SchoolHoliday**: Indicates school closure.
- **StoreType**: Types of stores.
- **Assortment**: Assortment levels.
- **CompetitionDistance**: Distance to the nearest competitor.
- **Promo**: Promo indicator.
- **Promo2**: Continuing promotions.
- **Promo2Since**: Start time of Promo2.
- **PromoInterval**: Promo intervals.

## Learning Outcomes
### Skills
- Advanced use of Scikit-learn
- Feature Engineering
- ML Model Building & Tuning
- CI/CD for ML Models
- Python Logging & Unit Testing
- Building Dashboards
- Model Management with DVC, CML, and MLFlow

### Knowledge
- Data Exploration & Predictive Analysis
- Machine Learning & Hyperparameter Tuning
- Communication: Reporting on Statistical Analysis

## Project Objectives
1. Data Exploration & Cleaning
2. Feature Engineering
3. ML and Deep Learning Model Building
4. Model Evaluation
5. Model Deployment & API Creation

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rossmann-sales-forecast.git
   cd rossmann-sales-forecast
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables (if applicable).

## Task Breakdown

### Task 1 - Exploration of Customer Purchasing Behavior
- Perform exploratory data analysis (EDA).
- Answer questions on purchasing behaviors:
  - Promo distribution in train/test sets.
  - Sales during holidays.
  - Correlation between sales and customers.
- **Key Deliverables**:
  - Visualizations (e.g., heatmaps, bar plots, line graphs).
  - Summary tables.
- **Logging**: Use Python's `logging` library for traceability.

### Task 2 - Prediction of Store Sales
#### 2.1 Preprocessing
- Handle missing values and outliers.
- Extract features from date columns (e.g., weekdays, holidays).
- Scale data using `StandardScaler`.

#### 2.2 Building Models with Sklearn Pipelines
- Use tree-based algorithms (e.g., Random Forest Regressor).
- Modularize with Sklearn Pipelines.

#### 2.3 Choose a Loss Function
- Justify the choice of a loss function (e.g., Mean Squared Error).

#### 2.4 Post Prediction Analysis
- Explore feature importance.
- Estimate confidence intervals.

#### 2.5 Serialize Models
- Save models with timestamped filenames (e.g., `model-YYYY-MM-DD-HH-MM.pkl`).

### Task 2.6 - Building Model with Deep Learning
- Transform time series data.
- Build an LSTM Regression model using TensorFlow or PyTorch.
- Ensure the model runs comfortably on Google Colab.

### Task 3 - Model Serving API Call
- **Framework**: Use FastAPI for REST API development.
- **Endpoints**:
  - `POST /predict`: Accepts input data and returns predictions.
- **Deployment**: Deploy on AWS/GCP/Heroku.

## File Structure
```
.github/workflows      # CI/CD workflows
fastapi-app/           # FastAPI application
logs/                  # Logging files/notebooks
notebooks/             # Jupyter notebooks for EDA and model building
scripts/               # Python scripts for preprocessing and modeling
tests/                 # Unit and integration tests
.gitignore             # Git ignore rules
Dockerfile             # Docker container configuration
README.md              # Project documentation
requirements.txt       # Python dependencies
```

## How to Use
1. Run the preprocessing pipeline:
   ```bash
   python scripts/preprocess.py
   ```
2. Train the model:
   ```bash
   python scripts/train_model.py
   ```
3. Start the API:
   ```bash
   uvicorn fastapi-app.main:app --reload
   ```
4. Test predictions:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"input": "sample data"}' http://127.0.0.1:8000/predict
   
