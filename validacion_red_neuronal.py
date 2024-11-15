# Importar librerías necesarias
from modelo.red_neuronal import red
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import precision_recall_curve

# Cargar la base de datos
data = pd.read_csv("./datos/diabetes.csv")

# Separar características (X) y etiqueta (y)
X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Aplicar SMOTE para balancear las clases
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Dividir los datos balanceados en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Inicializar la red neuronal con más neuronas en la capa oculta
mi_red = red(ncE=X.shape[1], ncO=32, ncS=1)  # Capa oculta con 32 neuronas

# Entrenamiento del modelo con minilotes de tamaño 10 y 1000 épocas
epochs = 1000
batch_size = 10
for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]

        for i in range(len(X_batch)):
            mi_red.entrena(X_batch[i], [y_batch[i]])

# Predicciones de salida en el conjunto de prueba
predicciones_prob = [mi_red.calculaSalida(x)[0] for x in X_test]

# Calcular precisión y recall para diferentes umbrales
precision, recall, thresholds = precision_recall_curve(y_test, predicciones_prob)

# Encontrar el umbral con la mejor combinación de precisión y recall
f1_scores = 2 * (precision * recall) / (precision + recall)
best_threshold = thresholds[np.argmax(f1_scores)]

# Predicciones finales con el mejor umbral
predicciones_finales = [1 if prob >= best_threshold else 0 for prob in predicciones_prob]

# Evaluación con el nuevo umbral
accuracy = accuracy_score(y_test, predicciones_finales)
print("Precision del modelo en datos de prueba:", accuracy)

print("Matriz de Confusion:")
print(confusion_matrix(y_test, predicciones_finales))

print("\nReporte de Clasificacion:")
print(classification_report(y_test, predicciones_finales))
