import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

iris = sns.load_dataset('iris')

sc = StandardScaler()
iris_without_species = iris.drop('species', axis=1)
sc.fit(iris_without_species)

scaled_data = sc.transform(iris_without_species)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# 2D Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=iris['species'], legend='full')
plt.title('PCA of Iris Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# 3D Plot
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(scaled_data)
fig = px.scatter_3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], color=iris['species'], labels={'x': 'First Principal Component', 'y': 'Second Principal Component', 'z': 'Third Principal Component'})
fig.show()