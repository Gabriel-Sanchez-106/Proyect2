import numpy as np

def moda(lst):
    return max(set(lst), key=lst.count)

def euclidiana(punto, datos):
    return np.sqrt(np.sum((punto - datos)**2, axis=1))

class KClasificadorVecinosCercanos:
    def __init__(self, k=10):
        self.k = k

    def entrenar(self, X_entrenamiento, y_entrenamiento):
        self.X_entrenamiento = X_entrenamiento
        self.y_entrenamiento = y_entrenamiento
  
    def predecir(self, X_prueba):
        vecinos = []
        for x in X_prueba:
            distancias = euclidiana(x, self.X_entrenamiento)
            y_estrella = [y for _, y in sorted(zip(distancias, self.y_entrenamiento))]
            vecinos.append(y_estrella[:self.k])
        return list(map(moda, vecinos))
    
    def puntaje(self, X_prueba, y_prueba):
        y_pred = self.predecir(X_prueba)
        precision = sum(y_pred == y_prueba) / len(y_prueba)
        return precision