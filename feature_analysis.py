import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# Load cleaned data
df = pd.read_csv('data/train_clean.csv')

# Basic dataset info
print("\n=== Cleaned Dataset Info ===")
print(f"Dataset Shape: {df.shape}")
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Structure:")
print(df.info())

# Feature Type Analysis
print("\n=== Feature Analysis ===")

# Numerical Features
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Age Distribution
sns.histplot(data=df, x='Age', bins=30, ax=ax1)
ax1.set_title('Age Distribution (Cleaned)')

# Fare Distribution  
sns.histplot(data=df, x='Fare', bins=30, ax=ax2)
ax2.set_title('Fare Distribution (Cleaned)')

# Categorical Features
sns.countplot(data=df, x='Sex', ax=ax3)
ax3.set_title('Gender Distribution (Cleaned)')

sns.countplot(data=df, x='Pclass', ax=ax4)
ax4.set_title('Passenger Class Distribution (Cleaned)')

plt.tight_layout()
plt.savefig('cleaned_feature_distributions.png')
plt.close()

# Survival Analysis
print("\n=== Survival Analysis ===")
survival_rate = df['Survived'].mean()
print(f"\nOverall Survival Rate: {survival_rate:.2%}")

# Survival by Class
class_survival = df.groupby('Pclass')['Survived'].mean()
print("\nSurvival Rate by Class:")
print(class_survival)

# Survival by Gender
gender_survival = df.groupby('Sex')['Survived'].mean()
print("\nSurvival Rate by Gender:")
print(gender_survival)

# Age Group Analysis
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
age_survival = df.groupby('AgeGroup')['Survived'].mean()
print("\nSurvival Rate by Age Group:")
print(age_survival)

# Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features (Cleaned)')
plt.tight_layout()
plt.savefig('cleaned_correlation_matrix.png')
plt.close()

# Categorical Feature vs Survival
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

sns.barplot(data=df, x='Pclass', y='Survived', ax=axes[0])
axes[0].set_title('Survival Rate by Passenger Class (Cleaned)')

sns.barplot(data=df, x='Sex', y='Survived', ax=axes[1])
axes[1].set_title('Survival Rate by Gender (Cleaned)')

sns.barplot(data=df, x='AgeGroup', y='Survived', ax=axes[2])
axes[2].set_title('Survival Rate by Age Group (Cleaned)')

sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[3])
axes[3].set_title('Fare Distribution by Survival (Cleaned)')

plt.tight_layout()
plt.savefig('cleaned_survival_analysis.png')
plt.close()

# Statistical Tests
print("\n=== Statistical Tests ===")

# Chi-square test for categorical variables
def chi_square_test(feature):
    contingency = pd.crosstab(df[feature], df['Survived'])
    chi2, p_value = stats.chi2_contingency(contingency)[:2]
    return feature, chi2, p_value

categorical_features = ['Pclass', 'Sex', 'Embarked']
print("\nChi-square tests for independence with survival:")
for feature in categorical_features:
    feature, chi2, p_value = chi_square_test(feature)
    print(f"{feature}: chi2={chi2:.2f}, p-value={p_value:.4f}")

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
    f.write("\n=== Survival Analysis ===\n")
    f.write(f"\nOverall Survival Rate: {survival_rate:.2%}\n")
    f.write("\nSurvival Rate by Class:\n")
    f.write(str(class_survival))
    f.write("\n\nGender-wise Survival Rates:\n")
    f.write(str(gender_survival))
    f.write("\n\nAge Group Survival Rates:\n")
    f.write(str(age_survival))
    
    # Statistical Tests
    f.write("\n\n=== Statistical Tests ===\n")
    f.write("\nChi-square tests for independence with survival:\n")
    for feature in categorical_features:
        feature, chi2, p_value = chi_square_test(feature)
        f.write(f"{feature}: chi2={chi2:.2f}, p-value={p_value:.4f}\n")
