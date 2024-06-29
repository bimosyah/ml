import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris')
sc = StandardScaler()
iris_without_species = iris.drop('species', axis=1)
sc.fit(iris_without_species)
scaled_data = sc.transform(iris_without_species)
reduce_data = LinearDiscriminantAnalysis(n_components=2).fit_transform(scaled_data, iris['species'])
plt.figure(figsize=(10, 6))
plt.title('LDA of Iris Dataset')
sns.scatterplot(x=reduce_data[:, 0], y=reduce_data[:, 1], hue=iris['species'], legend='full')
plt.xlabel('First Linear Discriminant')
plt.ylabel('Second Linear Discriminant')
plt.show()