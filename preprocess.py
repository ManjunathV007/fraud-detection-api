import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(credit_data):
    # Check Null values
    isNull = credit_data.isnull().sum()
    print("Null Value Check:")
    print(isNull)
    
    # Basic statistics
    print("\nDescription of data:")
    print(credit_data.describe())
    print("\nUnique Classes:", credit_data.Class.unique())
    
    # Check for duplicates
    duplicates = credit_data.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    return credit_data.drop_duplicates()

def scale_features(credit_data):
    scaler = StandardScaler()
    # Exclude 'Class' column from scaling
    columns_to_scale = [col for col in credit_data.columns if col != 'Class']
    credit_data[columns_to_scale] = scaler.fit_transform(credit_data[columns_to_scale])
    return credit_data

def detect_outliers(credit_data):
    outliers = {}
    for column in credit_data.columns:
        if column != 'Class':
            Q1 = credit_data[column].quantile(0.25)
            Q3 = credit_data[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers[column] = len(credit_data[(credit_data[column] < (Q1 - 1.5 * IQR)) | 
                                             (credit_data[column] > (Q3 + 1.5 * IQR))])
    print("\nOutliers per column:")
    print(pd.Series(outliers))
    return outliers

def visualize_data(credit_data):
    plt.figure(figsize=(15, 5))
    
    # Distribution of transaction amounts
    plt.subplot(1, 2, 1)
    sns.histplot(data=credit_data, x='Amount', hue='Class', bins=50)
    plt.title('Transaction Amount Distribution by Class')
    
    # Class distribution
    plt.subplot(1, 2, 2)
    sns.countplot(data=credit_data, x='Class')
    plt.title('Class Distribution')
    
    plt.tight_layout()
    plt.show()

def checkFraudVsValid(credit_data):
    fraud_case = credit_data[credit_data['Class'] == 1]
    valid_case = credit_data[credit_data['Class'] == 0]
    
    print("\nClass Balance:")
    print(f"Number of Fraud Cases: {len(fraud_case)}")
    print(f"Number of Valid Cases: {len(valid_case)}")
    print(f"Fraud Percentage: {(len(fraud_case)/len(credit_data))*100:.2f}%")
    
    return len(fraud_case), len(valid_case)

def balance_data(credit_data):
    X = credit_data.drop('Class', axis=1)
    y = credit_data['Class']
    
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    balanced_data = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), 
                             pd.Series(y_balanced, name='Class')], axis=1)
    
    print("\nAfter SMOTE balancing:")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Fraud cases: {len(balanced_data[balanced_data['Class'] == 1])}")
    print(f"Valid cases: {len(balanced_data[balanced_data['Class'] == 0])}")
    
    return balanced_data

def main(data_path):
    # Load data
    credit_data = pd.read_csv(data_path)
    
    # Preprocess
    credit_data = preprocess_data(credit_data)
    
    # Check class balance
    fraud_count, valid_count = checkFraudVsValid(credit_data)
    
    # Detect outliers
    outliers = detect_outliers(credit_data)
    
    # Visualize data
    visualize_data(credit_data)
    
    # Scale features
    credit_data = scale_features(credit_data)
    
    # Balance data if needed
    if fraud_count / valid_count < 0.4:  # threshold for imbalance
        credit_data = balance_data(credit_data)
    
    return credit_data

if __name__ == "__main__":
    processed_data = main("creditcard.csv")