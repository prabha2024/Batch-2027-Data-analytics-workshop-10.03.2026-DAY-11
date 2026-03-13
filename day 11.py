import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_classif

# -------------------------------
# 1. Load Iris dataset as list of lists
# -------------------------------

iris = load_iris()

data = iris.data.tolist()   # list of lists
target = iris.target
feature_names = iris.feature_names
species = iris.target_names

df = pd.DataFrame(iris.data, columns=feature_names)
df['species'] = target

print("First 5 rows of dataset:")
print(df.head())

# -------------------------------
# Mean and Standard Deviation for each column
# -------------------------------

print("\nMean of each column:")
print(df.iloc[:,0:4].mean())

print("\nStandard Deviation of each column:")
print(df.iloc[:,0:4].std())

# -------------------------------
# Mean and Std for each species
# -------------------------------

print("\nMean for each species:")
print(df.groupby('species').mean())

print("\nStandard deviation for each species:")
print(df.groupby('species').std())

# -------------------------------
# a. Box Plot (distribution)
# -------------------------------

plt.figure()
df.iloc[:,0:4].boxplot()
plt.title("Box Plot of Iris Measurements")
plt.show()

# -------------------------------
# Scatter Plot
# -------------------------------

plt.figure()
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Scatter Plot")
plt.show()

# -------------------------------
# b. Outliers using boxplot
# -------------------------------

plt.figure()
sns.boxplot(data=df.iloc[:,0:4])
plt.title("Outlier Detection using Box Plot")
plt.show()

# -------------------------------
# c. Histogram
# -------------------------------

df['sepal length (cm)'].hist()
plt.title("Histogram of Sepal Length")
plt.show()

# Bar Chart

species_count = df['species'].value_counts()

plt.figure()
species_count.plot(kind='bar')
plt.title("Bar Chart of Species Count")
plt.show()

# Pie Chart

plt.figure()
species_count.plot(kind='pie', autopct='%1.1f%%')
plt.title("Pie Chart of Species")
plt.show()

# -------------------------------
# d. Colorful Pie Chart
# -------------------------------

plt.figure()
plt.pie(species_count,
        labels=species,
        autopct='%1.1f%%',
        colors=['red','blue','green'])
plt.title("Colorful Pie Chart")
plt.show()

# -------------------------------
# 3D Histogram
# -------------------------------

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hist, xedges = np.histogram(df['sepal length (cm)'], bins=10)

xpos = xedges[:-1]
ypos = np.zeros(len(xpos))
zpos = np.zeros(len(xpos))

dx = 0.5
dy = 0.5
dz = hist

ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Y')
ax.set_zlabel('Frequency')

plt.title("3D Histogram")
plt.show()

# -------------------------------
# 2. Regression Example
# -------------------------------

X = df[['sepal length (cm)']]
y = df['petal length (cm)']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print("\nRegression R2 Score:", r2_score(y_test,pred))

plt.figure()
plt.scatter(X_test,y_test)
plt.plot(X_test,pred,color='red')
plt.title("Linear Regression")
plt.show()

# -------------------------------
# 3. Correlation Matrix
# -------------------------------

corr = df.iloc[:,0:4].corr()

print("\nCorrelation Matrix:")
print(corr)

# Correlation Plot

plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# ANOVA Test
# -------------------------------

F, p = f_classif(df.iloc[:,0:4], df['species'])

print("\nANOVA F values:", F)
print("ANOVA p values:", p)