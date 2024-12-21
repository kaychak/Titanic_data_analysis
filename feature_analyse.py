import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# Load cleaned data
df = pd.read_csv('data/train_clean.csv')

# Create unscaled version for comparison
df_unscaled = df.copy()
df_unscaled['Age'] = df['Age'] * df['Age'].std() + df['Age'].mean() 
df_unscaled['Fare'] = df['Fare'] * df['Fare'].std() + df['Fare'].mean()

# Correlation Matrix
plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlations')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Feature Relationships with Survival
# Box plots for numerical features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=df, x='Survived', y='Age', ax=axes[0])
axes[0].set_title('Age vs Survival')

sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[1])
axes[1].set_title('Fare vs Survival')

sns.boxplot(data=df, x='Survived', y='FamilySize', ax=axes[2])
axes[2].set_title('Family Size vs Survival')

plt.tight_layout()
plt.savefig('numerical_vs_survival.png')
plt.close()

# Bar plots for categorical features
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.barplot(data=df, x='Pclass', y='Survived', ax=axes[0])
axes[0].set_title('Survival Rate by Class')

sns.barplot(data=df, x='Sex', y='Survived', ax=axes[1])
axes[1].set_title('Survival Rate by Gender')

sns.barplot(data=df, x='Embarked', y='Survived', ax=axes[2])
axes[2].set_title('Survival Rate by Port')

plt.tight_layout()
plt.savefig('categorical_vs_survival.png')
plt.close()

# Feature Importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df.drop('Survived', axis=1), df['Survived'])

plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=df.drop('Survived', axis=1).columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Save summary statistics
with open('cleaned_eda_summary.txt', 'w') as f:
    f.write("=== Cleaned Titanic Dataset EDA Summary ===\n\n")
    
    # Dataset Info
    f.write("=== Dataset Info ===\n")
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("Dataset Structure:\n")
    buffer = io.StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue() + "\n")
    
    # Survival Analysis
    survival_rate = df['Survived'].mean()
    class_survival = df.groupby('Pclass')['Survived'].mean()
    gender_survival = df.groupby('Sex')['Survived'].mean()
    
    f.write("\n=== Survival Analysis ===\n")
    f.write(f"\nOverall Survival Rate: {survival_rate:.2%}\n")
    f.write("\nSurvival Rate by Class:\n")
    f.write(str(class_survival))
    f.write("\n\nGender-wise Survival Rates:\n")
    f.write(str(gender_survival))
    
    # Define chi_square_test function before using it
    def chi_square_test(feature):
        contingency = pd.crosstab(df[feature], df['Survived'])
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        return feature, chi2, p_value
    
    # Statistical Tests
    f.write("\n\n=== Statistical Tests ===\n")
    f.write("\nChi-square tests for independence with survival:\n")
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    for feature in categorical_features:
        feature, chi2, p_value = chi_square_test(feature)
        f.write(f"{feature}: chi2={chi2:.2f}, p-value={p_value:.4f}\n")
