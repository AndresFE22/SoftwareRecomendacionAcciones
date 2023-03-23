#IMPORTAR LIBRERIAS
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.ensemble  import RandomForestClassifier 

#IMPORTAR DATASET 
dataset = pd.read_csv('datasetPrediccionprueba.csv', delimiter= ";") 
#imprimir primeras 5 lineas del dataset 
print(dataset.head()) 

#Preparar datos para entrenar 
#Dividir atributos y etiquetas

X = dataset.iloc[:, 0:17].values
y = dataset.iloc[:, -1].values

#dividir los datos en etapas de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Crear random forest 
rfc=RandomForestClassifier(n_estimators=100)

rfc.fit(X_train,y_train) 

#Predicción 

Rango = rfc.predict([[4, 3, 1, 5, 1, 1, 1, 3, 1, 2, 1, 1, 2, 6, 8, 1, 1]])

#Mostrar predicción 

def showPrediction(argument):
   switcher = {
     1: "rf: Rango de inasistencia Alto",
     2: "rf: Rango de inasistencia Medio",
     3: "rf: Rango de inasistencia Bajo"
}

print (showPrediction(Rango[0]))
