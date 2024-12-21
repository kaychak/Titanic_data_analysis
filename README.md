# Titanic Survival Prediction Project

A machine learning project to predict passenger survival on the Titanic using multiple models and comprehensive analysis.

## Project Structure

- `eda.py`: Exploratory data analysis
- `data_analyse.py`: Data preprocessing and cleaning pipeline
- `feature_analyse.py`: Feature engineering and visualization
- `train_models.py`: Model training for logistic regression, decision tree, random forest, xgboost(with early stopping)
- `train_nostop.py`: Model training without early stopping for xgboost
- `predict.py`: Model prediction pipeline

## Features

### Data Preprocessing
- Missing value imputation
- Feature engineering (Title extraction, Family size, etc.)
- Categorical encoding
- Feature scaling

### Exploratory Data Analysis
- Missing value analysis
- Feature distributions
- Correlation analysis 
- Survival rate analysis by different features

### Models Implemented
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost (with early stopping)
- Deep Neural Network

### Model Training Features
- Learning curves visualization
- Feature importance analysis
- Cross-validation
- Early stopping (XGBoost only)
- Model performance metrics
- Model persistence with timestamps

## Usage

1. Data Preparation:
```python
analyzer = TitanicAnalyzer('data/train.csv')
cleaned_data = analyzer.clean_data()
```

2. Feature Analysis:
```python
# Run feature_analysis.py to generate visualizations and analysis
```

3. Train Models:
```python
trainer = TitanicModelTrainer('data/train_clean.csv')
model = trainer.train_xgboost()  # or other models
```

4. Make Predictions:
```python
predictor = Predictor('data/test.csv', 'preprocessor.joblib', 'model.joblib')
predictions = predictor.predict()
```

## Model Performance

Each model generates:
- Learning curves
- Loss curves
- Feature importance plots
- Classification reports
- Performance metrics

## Requirements

- python 3.11
- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow
- keras
- matplotlib
- seaborn
- joblib

## Files Generated

- Cleaned datasets
- Model artifacts (.joblib files)
- Performance visualizations
- Prediction results
- Analysis reports

## Best Practices Used

- Modular code structure
- Comprehensive documentation
- Visualization of model performance
- Model persistence with metadata
- Configurable preprocessing pipeline
- Robust error handling
