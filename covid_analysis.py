# 1. Importing Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset URL
url = "https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv"

# 1. Import Dataset
df = pd.read_csv(url)

# 2. High Level Data Understanding
print("Rows & Columns:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nData Info:\n")
df.info()
print("\nData Description:\n", df.describe(include='all'))

# 3. Low Level Data Understanding
print("\nUnique Locations:", df['location'].nunique())
print("\nContinent with Max Frequency:\n", df['continent'].value_counts().idxmax())
print("\nMax & Mean of Total Cases:", df['total_cases'].max(), df['total_cases'].mean())
print("\nQuartiles for Total Deaths:\n", df['total_deaths'].quantile([0.25, 0.5, 0.75]))
print("\nMax Human Development Index by Continent:\n", df.groupby('continent')['human_development_index'].mean().idxmax())
print("\nMin GDP per Capita by Continent:\n", df.groupby('continent')['gdp_per_capita'].mean().idxmin())

# 4. Filter Specific Columns
df = df[['continent', 'location', 'date', 'total_cases', 'total_deaths', 'gdp_per_capita', 'human_development_index']]

# 5. Data Cleaning
df = df.drop_duplicates()
print("\nMissing Values:\n", df.isnull().sum())
df = df.dropna(subset=['continent'])
df = df.fillna(0)

# 6. DateTime Formatting
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# 7. Data Aggregation by Continent
df_groupby = df.groupby('continent').max().reset_index()
print("\nGrouped Data:\n", df_groupby)

# 8. Feature Engineering
df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths'] / df_groupby['total_cases']

# 9. Data Visualization
sns.set(style="whitegrid")

# a. Histogram for gdp_per_capita
plt.figure(figsize=(8,5))
sns.histplot(df['gdp_per_capita'], kde=True)
plt.title('GDP per Capita Distribution')
plt.show()

# b. Scatter Plot
plt.figure(figsize=(8,5))
sns.scatterplot(x='total_cases', y='gdp_per_capita', data=df)
plt.title('Total Cases vs GDP per Capita')
plt.show()

# c. Pairplot
sns.pairplot(df_groupby)
plt.show()

# d. Bar plot: Continent vs Total Cases
sns.catplot(data=df_groupby, x='continent', y='total_cases', kind='bar')
plt.title('Total Cases by Continent')
plt.xticks(rotation=45)
plt.show()

# 10. Save the Final DataFrame
df_groupby.to_csv('df_groupby.csv', index=False)
print("File saved as 'df_groupby.csv'")
