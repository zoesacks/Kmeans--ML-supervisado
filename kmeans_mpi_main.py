import numpy as np
from PIL import Image, ImageEnhance
import os

from kmeans import KMeansPersonalizadoMPI

# Función para cargar, mejorar y preprocesar imágenes
# Carga imágenes de un directorio, ajusta brillo/contraste, añade fondo blanco si es PNG, las redimensiona y normaliza.
def cargar_y_mejorar_imagenes(ruta_directorio):
    imagenes_procesadas = []
    nombres_archivos = []  # Lista para almacenar los nombres de los archivos

    for archivo in os.listdir(ruta_directorio):
        if archivo.endswith(".png") or archivo.endswith(".jpg"):
            ruta_imagen = os.path.join(ruta_directorio, archivo)
            imagen = Image.open(ruta_imagen)

            # Si la imagen es PNG (puede tener transparencia), añadir fondo blanco
            if archivo.endswith(".png") and imagen.mode == 'RGBA':
                fondo_blanco = Image.new("RGBA", imagen.size, (255, 255, 255))
                imagen = Image.alpha_composite(fondo_blanco, imagen)
                imagen = imagen.convert('RGB')

            imagen = imagen.convert('L')
            imagen = ImageEnhance.Brightness(imagen).enhance(1.1)
            imagen = ImageEnhance.Contrast(imagen).enhance(1.1)
            imagen = imagen.resize((32, 32))

            imagen_vector = np.array(imagen).flatten()
            imagen_vector = imagen_vector / 255.0

            imagenes_procesadas.append(imagen_vector)
            nombres_archivos.append(archivo)  # Agregar nombre del archivo a la lista

    return np.array(imagenes_procesadas), nombres_archivos



# Ejemplo de uso
ruta_directorio = r"C:\Users\zsacks\Desktop\Kmeans- ML supervisado\imagenes"
data, nombres_archivos = cargar_y_mejorar_imagenes(ruta_directorio)
print(f"Imágenes procesadas: {data.shape}")

# Crear una instancia de KMeans personalizado
kmeans_personalizado = KMeansPersonalizadoMPI(n_clusters=10, max_iter=500, tol=1e-4, random_state=42)

# Ajustar el modelo y obtener las etiquetas de cada punto
labels = kmeans_personalizado.fit(data)

# Obtener nombres de archivos en todos los clusters
for cluster_id in range(kmeans_personalizado.n_clusters):
    archivos_en_cluster = kmeans_personalizado.obtener_nombres_en_cluster(labels, nombres_archivos, cluster_id)
    print(f"Archivos en el Cluster {cluster_id}: {archivos_en_cluster}")

# Visualizar los clusters
kmeans_personalizado.visualizar_clusters(data, labels)
 