import pandas as pd
import joblib
from datetime import datetime

# Load the Titanic dataset
df = pd.read_csv('data/train.csv')

class TitanicAnalyzer:
    def __init__(self, file_path=None):
        self.required_columns = [
            'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'
        ]
        self.normalization_params = {}
        self.categorical_mappings = {
            'Sex': {'male': 0, 'female': 1},
            'Embarked': {'S': 0, 'C': 1, 'Q': 2},
            'Title': {
                'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
                'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
                'Countess': 2, 'Ms': 1, 'Lady': 2, 'Jonkheer': 0,
                'Don': 0, 'Mme': 2, 'Capt': 4, 'Sir': 4
            },
            'Deck': {
                'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 
                'F': 5, 'G': 6, 'T': 7
            }
        }
        self.group_medians = {}  # Store medians for filling NaN values
        
        if file_path:
            self.data = pd.read_csv(file_path)
            self._verify_columns()
            self.output_file = file_path.replace('.csv', '_clean.csv')
            
    def _verify_columns(self):
        """Verify all required columns are present"""
        missing_cols = [col for col in self.required_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"❌ ERROR: Missing required columns: {', '.join(missing_cols)}")
        return True

    def fit(self, df):
        """Calculate and store all necessary transformation parameters"""
        self._verify_columns()  # Verify columns before fitting
        # Store group medians for Age and Fare
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False) #extract title from name
        df['Title'] = df['Title'].map(self.categorical_mappings['Title']) #map titles to numbers
        self.group_medians['Age'] = df.groupby('Title')['Age'].median().to_dict() #calculate median age for each title
        self.group_medians['Fare'] = df.groupby('Pclass')['Fare'].median().to_dict() #calculate median fare for each pclass
        
        # Calculate normalization parameters
        for feature in ['Age', 'Fare']:
            self.normalization_params[feature] = { #store mean and std for each feature
                'mean': df[feature].mean(),
                'std': df[feature].std()
            }
        
        return self

    def transform(self, df):
        """Transform raw data into model-ready features"""
        df = df.copy()
        
        # Store Survived column if it exists
        survived = df['Survived'] if 'Survived' in df.columns else None
        
        # Categorical mappings
        df['Sex'] = df['Sex'].map(self.categorical_mappings['Sex'])
        df['Embarked'] = df['Embarked'].map(self.categorical_mappings['Embarked'])
        df['Embarked'] = df['Embarked'].fillna(0)  # Default to most common 'S'
        
        # Extract and map Title
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].map(self.categorical_mappings['Title'])
        
        # Fill missing values using stored group medians
        for group_id, median_age in self.group_medians['Age'].items():
            df.loc[(df['Age'].isna()) & (df['Title'] == group_id), 'Age'] = median_age #fill missing age with median age for that title
        
        for pclass, median_fare in self.group_medians['Fare'].items():
            df.loc[(df['Fare'].isna()) & (df['Pclass'] == pclass), 'Fare'] = median_fare
        
        # Normalize numeric features
        for feature, params in self.normalization_params.items():
            df[feature] = (df[feature] - params['mean']) / params['std'] #z-score normalization of numeric features
        
        # Create family size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 #create family size feature
        
        # Extract and map Deck
        df['Deck'] = df['Cabin'].str[0] #extract deck from cabin
        df['Deck'] = df['Deck'].map(self.categorical_mappings['Deck'])
        df['Deck'] = df['Deck'].fillna(-1)
        
        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        df = df.drop(columns_to_drop, axis=1)
        
        # Ensure fixed column order
        fixed_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'Deck']
        
        # Add Survived back if it existed
        if survived is not None:
            df['Survived'] = survived
            fixed_columns = ['Survived'] + fixed_columns
        
        df = df[fixed_columns]
        
        return df

    def fit_transform(self, df):
        """Convenience method to fit and transform in one step"""
        return self.fit(df).transform(df) #fit calculates the parameters, transform uses them to clean the data

    def clean_data(self):
        """Legacy method that now uses fit_transform"""
        cleaned_df = self.fit_transform(self.data)
        cleaned_df.to_csv(self.output_file, index=False)
        return cleaned_df

    def analyze_features(self):
        """Analyze features in the dataset"""
        analysis = {
            'missing_values': self.data.isnull().sum(),
            'value_counts': {
                col: self.data[col].value_counts() 
                for col in self.data.columns
            }
        }
        return analysis

if __name__ == "__main__":
    analyzer = TitanicAnalyzer('data/train.csv')
    cleaned_data = analyzer.clean_data()
    current_date = datetime.now().strftime('%Y%m%d')
    joblib.dump(analyzer, f'titanic_analyzer_{current_date}.joblib')
    print(cleaned_data.head())  