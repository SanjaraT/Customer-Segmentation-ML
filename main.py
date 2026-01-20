import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv("Mall_Customers.csv")
# print(df.head())
# print(df.shape)
# print(df.info())

# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())

#Distribution
num_cols = ['CustomerID','Age','AnnualIncome','SpendingScore']
# for col in num_cols:
#     sns.histplot(df[col], kde=True)
#     plt.title("Distribution")
    # plt.show()

#Boxplots
# for col in num_cols:
#     sns.boxplot(x=df[col])
#     plt.title(f'Boxplot of {col}')
    # plt.show()

#Pairplots
# sns.pairplot(df[num_cols])
# plt.show()

#PREPROCESSING
df.drop(columns=['CustomerID'])
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
X = df[['Age', 'AnnualIncome', 'SpendingScore']]

#scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#elbow method(k=?)
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# plt.plot(range(1, 11), wcss, marker='o')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.title('Elbow Method')
# plt.show()

#KMEANS CLUSTERING
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

sns.scatterplot(
    x='AnnualIncome',
    y='SpendingScore',
    hue='cluster',
    data=df,
    palette='viridis'
)
plt.title("Customer Segments based on Income and Spending")
plt.show()



