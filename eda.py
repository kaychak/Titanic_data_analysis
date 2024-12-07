import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import io

# Load data
df = pd.read_csv('data/train.csv')

# Basic dataset info
print("\n=== Dataset Info ===")
print(f"Dataset Shape: {df.shape}")
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDataset Structure:")
print(df.info())

# 1. Missing Value Analysis
plt.figure(figsize=(10, 6))
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (missing_data / len(df) * 100).round(2)

sns.barplot(x=missing_percent.index, y=missing_percent.values)
plt.title('Percentage of Missing Values by Feature')
plt.xticks(rotation=45)
plt.ylabel('Missing Percentage')
plt.tight_layout()
plt.savefig('missing_values.png')
plt.close()

# 2. Numerical Features Distribution
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(numerical_features):
    # Histogram
    sns.histplot(data=df, x=feature, bins=30, ax=axes[idx], kde=True)
    axes[idx].set_title(f'{feature} Distribution')
    
    # Add boxplot as an inset
    inset_ax = axes[idx].inset_axes([0.6, 0.6, 0.35, 0.35])
    sns.boxplot(data=df, x=feature, ax=inset_ax)
    inset_ax.set_title('Boxplot')

plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.close()

# 3. Categorical Features Distribution
categorical_features = ['Pclass', 'Sex', 'Embarked', 'SibSp']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(categorical_features):
    # Count plot
    sns.countplot(data=df, x=feature, ax=axes[idx])
    axes[idx].set_title(f'{feature} Distribution')
    
    # Add percentage labels
    total = len(df[feature])
    for p in axes[idx].patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        axes[idx].annotate(percentage, (x, y), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('categorical_distributions.png')
plt.close()

# Extract titles from Name column
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Calculate survival statistics before writing to file
survival_rate = df['Survived'].mean()
class_survival = df.groupby('Pclass')['Survived'].mean()
gender_survival = df.groupby('Sex')['Survived'].mean()

# Create age groups and calculate survival rates
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
age_survival = df.groupby('AgeGroup')['Survived'].mean()

# Add this function before the statistical tests section
def chi_square_test(feature):
    contingency = pd.crosstab(df[feature], df['Survived'])
    chi2, p_value, _, _ = stats.chi2_contingency(contingency)
    return feature, chi2, p_value

# Save summary statistics
with open('eda_summary.txt', 'w') as f:
    f.write("=== Titanic Dataset EDA Summary ===\n\n")
    
    # Dataset Info
    f.write("=== Dataset Info ===\n")
    f.write(f"Dataset Shape: {df.shape}\n\n")
    f.write("Missing Values:\n")
    f.write(str(df.isnull().sum()) + "\n\n")
    f.write("Dataset Structure:\n")
    buffer = io.StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue() + "\n")
    
    # Feature Analysis
    f.write("=== Feature Analysis ===\n")
    f.write("\nTitle Distribution:\n")
    f.write(str(df['Title'].value_counts()) + "\n")
    
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
