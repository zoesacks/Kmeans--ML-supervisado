import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import os

# Función para cargar, mejorar y preprocesar imágenes
# Carga imágenes de un directorio, ajusta brillo/contraste, añade fondo blanco si es PNG, las redimensiona y normaliza.
def cargar_y_mejorar_imagenes(ruta_directorio):
    imagenes_procesadas = []
    for archivo in os.listdir(ruta_directorio):
        if archivo.endswith(".png") or archivo.endswith(".jpg"):
            ruta_imagen = os.path.join(ruta_directorio, archivo)
            imagen = Image.open(ruta_imagen)

            # Si la imagen es PNG (puede tener transparencia), añadir fondo blanco
            if archivo.endswith(".png") and imagen.mode == 'RGBA':
                fondo_blanco = Image.new("RGBA", imagen.size, (255, 255, 255))      # Crear una imagen con fondo blanco
                imagen = Image.alpha_composite(fondo_blanco, imagen)                # Combinar la imagen original con el fondo blanco usando la máscara alfa
                imagen = imagen.convert('RGB')                                      # Convertir de RGBA a RGB (sin canal alfa)

            imagen = imagen.convert('L')                                            # Convertir la imagen a escala de grises
            imagen = ImageEnhance.Brightness(imagen).enhance(1.1)                   # Mejorar ligeramente el brillo
            imagen = ImageEnhance.Contrast(imagen).enhance(1.1)                     # Mejorar ligeramente el contraste
            imagen = imagen.resize((32, 32))                                        # Redimensionar a un tamaño

            imagen_vector = np.array(imagen).flatten()
            imagen_vector = imagen_vector / 255.0                                   # Normalizar entre 0 y 1

            imagenes_procesadas.append(imagen_vector)

    return np.array(imagenes_procesadas)



ruta_directorio = r"C:\Users\zsacks\Desktop\Kmeans- ML supervisado\imagenes"        # Ruta del directorio con imágenes para procesamiento 
data = cargar_y_mejorar_imagenes(ruta_directorio)                                   # Llamar a la función para cargar y preprocesar las imágenes
print(f"Imágenes procesadas: {data.shape}")

n = 10                     # Número de clusters

# Ejecutar KMeans con parámetros ajustados
kmeans = KMeans(
    n_clusters=n,          # Elegir el número de clusters
    init='random',      # Inicialización inteligente de los centroides. Se puede usar 'random'
    n_init=10000,          # Ejecutar 10,000 veces con diferentes inicializaciones
    max_iter=500,          # Permitir más iteraciones para mejor convergencia
    tol=1e-4,              # Mantener el criterio de convergencia por defecto
    random_state=None,     # Usar semilla para reproducibilidad, o None para aleatoriedad
    algorithm='lloyd',     # Usa el algoritmo clásico de k-means. Otras opciones: 'elkan' para eficiencia
    verbose=0              # Mantener salida en silencio
)

# Ajustar y predecir los clusters
kmeans.fit(data)
Z = kmeans.predict(data)

# Visualizar los resultados por cluster
for i in range(0, n):
    fila = np.where(Z == i)[0]          # Filas donde están las imágenes de cada cluster
    num = fila.shape[0]                 # Número de imágenes de cada cluster
    r = int(np.floor(num / 10.))        # Número de filas en la figura de salida

    print("Cluster " + str(i))
    print(str(num) + " elementos")

    # Dibujar las imágenes de cada cluster
    plt.figure(figsize=(10, 10))
    
    for k in range(0, num):
        plt.subplot(r + 1, 10, k + 1)
        imagen = data[fila[k], ]
        imagen = imagen.reshape(32, 32)  # Asegurarse de que las imágenes estén en 32x32
        plt.imshow(imagen, cmap=plt.cm.gray)
        plt.axis('off')

    plt.show()
 