import pandas as pd
import numpy as np
#cargar el conjunto de datos desde un archivo csv
dataset = pd.read_csv("datasetPrediccionprueba.csv",delimiter=";")
from sklearn.model_selection import train_test_split
#linea para importar el MLPClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns # Convention alias for Seaborn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from array import array
# Import necessary modules
from sklearn.model_selection import train_test_split
print(dataset.head())

X = dataset.iloc[:,0:18].values
y = dataset.iloc[:,-1].values

#error en train_test_split: se debe importar con from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Hacer la red neuronal
#@title Texto de título predeterminado
iactivation = "logistic" #@param ["relu", "identity", "logistic", "softmax"]
isolver = "lbfgs" #@param ["adam", "lbfgs"]
imaxiter = 1000 #@param {type:"integer"}



mi_red_neuronal= MLPClassifier(activation=iactivation,
                               random_state=1,
                               solver= isolver,
                               max_iter=imaxiter).fit(X_train, y_train)

from seaborn.matrix import heatmap


#@title Texto de título predeterminado
iescuela = 1 #@param {type:"number"}
trabajas = 1 #@param {type:"number"}
vision = 2 #@param {type:"number"}

gastotra = 1 #@param {type:"number"}
tiempoce = 1 #@param {type:"number"}
becaestu = 2 #@param {type:"number"}
califrec = 4 #@param {type:"number"}
pisocasa = 3 #@param {type:"number"}
aguaserv = 1 #@param {type:"number"}
luzelect = 2 #@param {type:"number"}
internet = 1 #@param {type:"number"}
vivencas = 2 #@param {type:"number"}
trabajop = 3 #@param {type:"number"}
cvivestu = 5 #@param {type:"number"}
jefecasa = 3 #@param {type:"number"}
jefeesco = 1 #@param {type:"number"}
saludafi = 1 #@param {type:"number"}
estratos = 4 #@param {type:"number"}



newcase =    [iescuela,
                                  trabajas,
                                  vision,
                                  gastotra,
                                  tiempoce,
                                  becaestu,
                                  califrec,
                                  pisocasa,
                                  aguaserv,
                                  luzelect,
                                  internet,
                                  vivencas,
                                  trabajop,
                                  cvivestu,
                                  jefecasa,
                                  jefeesco,
                                  saludafi,
                                  estratos]

rango = mi_red_neuronal.predict([[iescuela,
                                  trabajas,
                                  vision,
                                  gastotra,
                                  tiempoce,
                                  becaestu,
                                  califrec,
                                  pisocasa,
                                  aguaserv,
                                  luzelect,
                                  internet,
                                  vivencas,
                                  trabajop,
                                  cvivestu,
                                  jefecasa,
                                  jefeesco,
                                  saludafi,
                                  estratos]])


print(mi_red_neuronal.score(X_test, y_test))

y_pred = mi_red_neuronal.predict(X_test)

respuestaredneuronalo = rango[0]

mensajeprediccion =  "No se puedo determinar la respuesta de la prediccion"
if (respuestaredneuronalo == 0):
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia bajo"
elif (respuestaredneuronalo == 1): 
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia medio"
elif (respuestaredneuronalo == 2): 
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia Alto"



print(mi_red_neuronal.score(X_test, y_test))
print(mensajeprediccion)
score = mi_red_neuronal.score(X_test, y_test)

variables = ['iescuela', 'trabajas', 'vision','gastotra','tiempoce', 'becaestu', 'califrec', 'pisocasa','aguaserv','luzelect','internet', 'vivencas', 'trabajop','cvivestu','jefecasa', 'jefeesco','saludafi','estratos']


for var in variables:
    plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='rangoina', data=dataset).set(title=f'Regression plot of {var} and inasistencia');
correlations = dataset.corr()
#annot=True #displays the correlation values
sns.heatmap(correlations, annot=True).set(title='Heatmap of Ton Data - Pearson Correlations');


# Getting the Pearson Correlation Coefficient
#print("Pearson Correlation of two features")
#print(correlations.loc['escuela', 'rangoina'])
fieldtomodify = ""
listacampo = []
lista = []
for var in variables:
    #plt.figure() # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    #sns.regplot(x=var, y='Ton', data=dataset).set(title=f'Regression plot of {var} and Ton');
    
    print("Pearson Correlation between -> %s and %s = %s" % (var, 'rangoina',correlations.loc[var, 'rangoina'] ))
    fieldtomodify = var
    lista.append(correlations.loc[var, 'rangoina'])
    listacampo.append((var,correlations.loc[var, 'rangoina']))

def obtener_nombre_campo(lista_campos_y_valores, valor_buscado):
    for tupla in lista_campos_y_valores:
        if tupla[1] == valor_buscado:
            nombre_campo = tupla[0]
            return nombre_campo
    return None

lista.sort(reverse=True)

def obtener_indice_tupla(lista_campos_y_valores, nombre_campo_buscado):
    for indice, tupla in enumerate(lista_campos_y_valores):
        if tupla[0] == nombre_campo_buscado:
            return indice
    return None

print("esta es la lista")
print(listacampo)

maximo = max(lista)
print("indice")

campomayor = obtener_nombre_campo(listacampo, maximo)
indice_maximo_valor = obtener_indice_tupla(listacampo, campomayor)
print("El valor a modificar en el caso nuevo es")
print(newcase[indice_maximo_valor])

print(indice_maximo_valor)
#fieldtomodify = variables[indice_maximo_valor]
print("Este es el campo modificado")
print(fieldtomodify)
minimo = min(lista)

print("Valor maximo: ")
print(maximo)
print("Valor minimo: ")
print(minimo)
print("este es el dataset corr")
print(dataset.corr)
corr_matrix = dataset.corr()
print("Este es el nombre del campo que hay que modificar")
print(obtener_nombre_campo(listacampo, maximo)) 





column_list = corr_matrix.columns.tolist()

print(column_list)

#Aqui se hace la simulación de la prediccion con la modificacion del campo de mayor y de menor valor

#Configurar el nuevo caso ajustando el indice del maximo valor

valororiginal = newcase[indice_maximo_valor]
#modificacion del valor original

#Se le añade - para aclarar que no tiene ningun valor
nuevovalor = -1 

if maximo > 0: 
  nuevovalor = valororiginal - 1
else :  nuevovalor = valororiginal + 1

#Reconfigurando el case

newcase[indice_maximo_valor] = nuevovalor

#Haciendo la simulacion de la prediccion

simulacion = mi_red_neuronal.predict([newcase])

respuestaredneuronals = rango[0]
print("Estos los datos ajustado con la inteligencia artificial de la simulacion")

mensajeprediccion =  "No se puedo determinar la respuesta de la prediccion"
if (respuestaredneuronals == 0):
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia bajo"
elif (respuestaredneuronals == 1): 
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia medio"
elif (respuestaredneuronals == 2): 
  mensajeprediccion = "El estudiante puede tener un rango de inasistencia Alto"

print(simulacion)
print(mensajeprediccion)

if (respuestaredneuronals > respuestaredneuronalo):
  print("la simulacion es mayor que la original")
  



else:
  print("No cambio el resultado")
