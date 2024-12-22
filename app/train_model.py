import joblib
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import os


model_folder='app/models'

# Crear la carpeta 'models' si no existe
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

# Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Las características de las flores (no hay etiquetas en este caso)

# Crear el modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)  # Número de clusters (3, porque sabemos que hay 3 especies)

# Entrenar el modelo con el dataset (solo se usa X, no se necesita y porque K-Means es no supervisado)
kmeans.fit(X)

# Guardar el modelo en la carpeta 'models' como 'kmeans_model.pkl'
joblib.dump(kmeans, f'{model_folder}/kmeans_model.pkl')

print("Modelo KMeans entrenado y guardado exitosamente en 'models/kmeans_model.pkl'")

X = iris.data
y = iris.target

# Crear el modelo KNN
model = KNeighborsClassifier(n_neighbors=3)

# Entrenar el modelo con el dataset
model.fit(X, y)

# Guardar el modelo en la carpeta 'models' como 'knn_model.pkl'
joblib.dump(model, f'{model_folder}/knn_model.pkl')

print("Modelo KNN entrenado y guardado exitosamente en 'models/knn_model.pkl'")
