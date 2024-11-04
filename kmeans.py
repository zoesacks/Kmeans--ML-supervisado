from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

class KMeansPersonalizadoMPI:
    def __init__(self, n_clusters=10, max_iter=500, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters    # numero de clusters
        self.max_iter = max_iter        # max de iteraciones
        self.tol = tol                  # valor de tolerancia, para que no procese de mas
        self.random_state = random_state # semilla el random
        self.centroides = None          # centroides que en un principio no hay

    # inicializar centroides de forma aleatoria
    def inicializar_centroides(self, data):
        if self.random_state:
            np.random.seed(self.random_state)   # iniciar el random con la semilla
        indices = np.random.choice(data.shape[0], self.n_clusters, replace=False)   # seleccionar los centroides en bases al random
        self.centroides = data[indices]
        
    # asignar cada punto al centroide más cercano
    def asignar_clusters(self, data_chunk):
        # para calcular la distancia euclidiana transforma la imagen a vector de esta forma se puede calcular la diferencia entre ambos
        distancias = np.sqrt((
            (data_chunk - self.centroides[:, np.newaxis]) # resta cada centroide de cada imagen
            ** 2).sum(axis=2))                            # eleva al cuadrado las diferencias y suma
        
        return np.argmin(distancias, axis=0)              # argmin selecciona el más cercano
    
    # actualizar los centroides segun el promedio de los puntos asignados
    def actualizar_centroides(self, data, labels):
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    # ajustar el modelo KMeans con MPI
    def fit(self, data):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Solo el proceso 0 inicializa los centroides
        if rank == 0:
            self.inicializar_centroides(data)  # inicializa centroides
        else:
            self.centroides = np.empty((self.n_clusters, data.shape[1]), dtype=data.dtype)
        
        # Broadcast de los centroides a todos los procesos
        comm.Bcast(self.centroides, root=0)
        
        for i in range(self.max_iter):
            # Scatter: distribuir porciones de datos a cada proceso
            data_chunk = np.array_split(data, size)[rank]
            
            # Cada proceso asigna los clusters a su porción de datos
            local_labels = self.asignar_clusters(data_chunk)  # asigna el cluster para el segmento local
            
            # Recoger todas las etiquetas de cada proceso en el proceso 0
            labels = comm.gather(local_labels, root=0)

            # Solo el proceso 0 actualiza los centroides
            if rank == 0:
                labels = np.concatenate(labels)  # Combinar etiquetas de todos los procesos
                nuevos_centroides = self.actualizar_centroides(data, labels)  # actualizar centroides

                # Calcular si los nuevos centroides son similares a los anteriores
                if np.all(np.abs(nuevos_centroides - self.centroides) < self.tol):
                    break
                self.centroides = nuevos_centroides
            
            # Broadcast de los nuevos centroides a todos los procesos
            comm.Bcast(self.centroides, root=0)

        # Solo el proceso 0 retorna las etiquetas finales
        if rank == 0:
            return labels

    # predecir el cluster de nuevos puntos
    def predict(self, data):
        return self.asignar_clusters(data)
    
    # visualizar los clusters
    def visualizar_clusters(self, data, labels, image_shape=(32, 32)):
        for i in range(self.n_clusters):
            fila = np.where(labels == i)[0]
            num = fila.shape[0]
            r = int(np.floor(num / 10.))
            
            print("Cluster " + str(i))
            print(str(num) + " elementos")

            plt.figure(figsize=(10, 10))
            for k in range(num):
                plt.subplot(r + 1, 10, k + 1)
                imagen = data[fila[k]]
                imagen = imagen.reshape(image_shape)
                plt.imshow(imagen, cmap=plt.cm.gray)
                plt.axis('off')

            plt.show()


    def obtener_nombres_en_cluster(self, labels, nombres_archivos, cluster_id):
        """
        Devuelve los nombres de los archivos en un cluster específico.
        
        :param labels: Etiquetas de clustering para cada dato
        :param nombres_archivos: Lista con los nombres de los archivos correspondientes a los datos
        :param cluster_id: El ID del cluster para el cual queremos obtener los nombres de archivos
        :return: Lista de nombres de archivos en el cluster especificado
        """
        archivos_en_cluster = [nombres_archivos[i] for i in range(len(labels)) if labels[i] == cluster_id]
        return archivos_en_cluster
