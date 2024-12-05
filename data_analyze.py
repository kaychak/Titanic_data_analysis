import pandas as pd

# Load the Titanic dataset
df = pd.read_csv('data/train.csv')

class TitanicAnalyzer:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.output_file = file_path.replace('.csv', '_clean.csv')
        self.normalization_params = {}  # Store normalization parameters
        
    def analyze_features(self):
        """Analyze each feature's type and distribution"""
        analysis = {}
        for col in self.data.columns:
            col_type = self.data[col].dtype
            unique_vals = len(self.data[col].unique())
            missing = self.data[col].isnull().sum()
            
            if col_type in ['int64', 'float64']:
                analysis[col] = {
                    'type': 'numeric',
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'missing': missing
                }
            elif unique_vals < 10:
                analysis[col] = {
                    'type': 'categorical',
                    'categories': self.data[col].value_counts().to_dict(),
                    'missing': missing
                }
            else:
                analysis[col] = {
                    'type': 'string/other',
                    'unique_values': unique_vals,
                    'missing': missing
                }
                
        return analysis
    
    def clean_data(self):
        """Clean and transform features"""
        df = self.data.copy()
        
        # Handle categorical features
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
        # Extract title from Name
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
            'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
            'Countess': 2, 'Ms': 1, 'Lady': 2, 'Jonkheer': 0,
            'Don': 0, 'Mme': 2, 'Capt': 4, 'Sir': 4
        }
        df['Title'] = df['Title'].map(title_mapping)
        
        # Handle numeric features
        df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('median'))
        df['Fare'] = df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'))
        
        # Normalize numeric features
        numeric_features = ['Age', 'Fare']
        for feature in numeric_features:
            mean = df[feature].mean()
            std = df[feature].std()
            self.normalization_params[feature] = {'mean': mean, 'std': std}
            df[feature] = (df[feature] - mean) / std
            
        # Create family size feature
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Extract deck from Cabin
        df['Deck'] = df['Cabin'].str[0]
        deck_mapping = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 
            'F': 5, 'G': 6, 'T': 7
        }
        df['Deck'] = df['Deck'].map(deck_mapping)
        df['Deck'] = df['Deck'].fillna(-1)  # Missing cabins
        
        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        df = df.drop(columns_to_drop, axis=1)
        
        # Save cleaned data
        df.to_csv(self.output_file, index=False)
        return df

    def normalize_new_data(self, new_data):
        """Apply saved normalization to new data"""
        df = new_data.copy()
        for feature, params in self.normalization_params.items():
            df[feature] = (df[feature] - params['mean']) / params['std']
        return df

analyzer = TitanicAnalyzer('data/train.csv')
feature_analysis = analyzer.analyze_features()
cleaned_data = analyzer.clean_data()
