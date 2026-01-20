import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt


df = pd.read_csv("Mall_Customers.csv")
# print(df.head())
# print(df.shape)
# print(df.info())

# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())

#Distribution
num_cols = ['CustomerID','Age','AnnualIncome','SpendingScore']
for col in num_cols:
    sns.histplot(df[col], kde=True)
    plt.title("Distribution")
    # plt.show()

#Boxplots
for col in num_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    # plt.show()

#Pairplots
sns.pairplot(df[num_cols])
plt.show()