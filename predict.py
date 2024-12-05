import pandas as pd
import joblib
from datetime import datetime
import os
from keras.models import load_model
from data_analyze import TitanicAnalyzer

class Predictor:
    def __init__(self, data, preprocessor, model):
        self.data = data
        self.preprocessor : TitanicAnalyzer = joblib.load(preprocessor)
        
        # Load model based on file extension
        if model.endswith('.h5'):
            self.model = load_model(model)
        else:
            loaded_model = joblib.load(model)
            if isinstance(loaded_model, dict):
                # Handle the case where the model is a dictionary
                self.model = loaded_model.get('model')  # Adjust key as needed
            else:
                self.model = loaded_model
            
        self.model_name = os.path.basename(model).split('.')[0].split('_')[0]

    def preprocess_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data)
        X = self.preprocessor.fit_transform(df)
        return X, df['PassengerId']

    def predict(self):
        X, passenger_ids = self.preprocess_data()
        
        # Make predictions
        predictions = self.model.predict(X)
        # Flatten only for deep learning models
        if self.model_name.startswith('deep'):
            predictions = (predictions > 0.5).astype(int).flatten()
        else:
            predictions = (predictions > 0.5).astype(int) if len(predictions.shape) > 1 else predictions
        
        results_df = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        
        # Save predictions
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'predictions_{self.model_name}_{timestamp}.csv', index=False)
        
        # Generate report
        total_passengers = len(predictions)
        survivors = predictions.sum()
        survival_ratio = survivors / total_passengers
        
        with open(f'{self.model_name}_{timestamp}_ratio.txt', 'w') as f:
            f.write(f'Total Passengers: {total_passengers}\n')
            f.write(f'Survivors: {survivors}\n')
            f.write(f'Survival Ratio: {survival_ratio:.2%}\n')
            
        return results_df

if __name__ == "__main__":
    data = 'data/test.csv'
    preprocessor = 'titanic_analyzer_20241205.joblib'
    model = 'deep_result_3/deep_learning_model_20241205_101657_0.7933.h5'
    predictor = Predictor(data, preprocessor, model)
    predictor.predict()