import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

iris = sns.load_dataset('iris')
sc = StandardScaler()
iris_without_species = iris.drop('species', axis=1)
sc.fit(iris_without_species)
scaled_data = sc.transform(iris_without_species)
tsne = TSNE(n_components=2, max_iter=1000, random_state=42)
reduced_data = tsne.fit_transform(scaled_data)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=iris['species'], legend='full')
plt.xlabel('First t-SNE Feature')
plt.ylabel('Second t-SNE Feature')
plt.title('t-SNE of Iris Dataset')
plt.show()
