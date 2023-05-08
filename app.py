from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
#cargar el conjunto de datos desde un archivo csv
from sklearn.model_selection import train_test_split
#linea para importar el MLPClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns # Convention alias for Seaborn
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

#import matplotlib 
#import matplotlib.pyplot as plt 
#import seaborn as sns 
dataset = pd.read_csv('datasetPrediccionprueba.csv', delimiter=';')
df = pd.read_csv('recom.csv', delimiter=';', encoding='iso-8859-1')
dfh = pd.read_csv('heuristica.csv', delimiter=';', encoding='iso-8859-1')

print(dataset.head())
print(df.columns)
print(df.head())
print(dfh.head())

app = Flask(__name__) 
app.secret_key = 'mi_clave_secreta'

@app.route('/')
def index():
    return render_template('indexinter.html')

@app.route('/showpagetraining')
def showpagetraining():
  #imprimir primeras 5 lineas del dataset 
  print(dataset.head()) 

  #Preparar datos para entrenar 
  #Dividir atributos y etiquetas
  X = dataset.iloc[:, 0:18].values
  y = dataset.iloc[:, -1].values



#dividir los datos en etapas de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Algoritmo de red neuronal
  iactivation = "logistic"
  isolver = "lbfgs"
  imaxiter = 1000

  mi_red_neuronal= MLPClassifier(activation=iactivation,
                                random_state=1,
                                solver= isolver,
                                max_iter=imaxiter).fit(X_train, y_train)
  

  return render_template('training.html')

@app.route('/showpageprediction')
def showpageprediction():
  
  return render_template('formprediction.html')
  


@app.route('/doprediction', methods=['POST'])
def doprediction():

 #Preparar datos para entrenar 
  #Dividir atributos y etiquetas
  X = dataset.iloc[:, 0:18].values
  y = dataset.iloc[:, -1].values


#dividir los datos en etapas de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Algoritmo de red neuronal
  iactivation = "logistic"
  isolver = "lbfgs"
  imaxiter = 1000

  mi_red_neuronal= MLPClassifier(activation=iactivation,
                                random_state=1,
                                solver= isolver,
                                max_iter=imaxiter).fit(X_train, y_train)

#global showPrediction 
  iescuela = int(request.form['iescuela'])
  #print ("escuela", iescuela)
  trabajas = int(request.form['trabajas'])
  #python no lee x y ñ
  vision = int(request.form['vision'])
  #print ("trabajo", trabajas)
  gastotra = int(request.form['gastotra'])
  #print ("gastos", gastotra)
  tiempoce = int(request.form['tiempoce'])
  #print ("tiempo", tiempoce)
  becaestu = int(request.form['becaestu'])
  #print ("becas", becaestu)
  califrec = int(request.form['califrec'])
  #print ("calificacion", califrec)
  pisocasa = int(request.form['pisocasa'])
  #print ("piso", pisocasa)
  aguaserv = int(request.form['aguaserv'])
  #print ("agua", aguaserv)  
  luzelect = int(request.form['luzelect'])
  #print ("luz", luzelect)
  internet = int(request.form['internet'])
  #print ("internet", internet)
  vivencas = int(request.form['vivencas'])
  #print ("viven", vivencas)
  trabajop = int(request.form['trabajop'])
  #print ("trabajo", trabajop)
  vivescon = int(request.form['vivescon'])
  #print ("vives", vivescon)
  escojefe = int(request.form['escojefe'])
  #print ("escolaridad", escojefe)
  parenjef = int(request.form['parenjef'])
  #print ("parentezco", parenjef)
  saludafi = int(request.form['saludafi'])
  #print ("salud", saludafi)
  estratos = int(request.form['estratos'])
  #print ("estrato", estratos)
  #return "<h1>Bienvenido " + iescuela + "</h1>"
  #return ('Hello, Worldx!', iescuela)
  #return render_template('showprediction.html', variable= estratos)


#Predicción 

  rango = mi_red_neuronal.predict([[iescuela,trabajas,vision,gastotra,tiempoce,becaestu,califrec,pisocasa,aguaserv,luzelect,internet,vivencas,trabajop,vivescon,parenjef,escojefe,saludafi,estratos]])
#Nueva prediccion
  newcase = [iescuela,trabajas,vision,gastotra,tiempoce,becaestu,califrec,pisocasa,aguaserv,luzelect,internet,vivencas,trabajop,vivescon,parenjef,escojefe,saludafi,estratos]
  

  #prediccion=showPrediction(rango[0])
  y_pred = mi_red_neuronal.predict(X_test)
  respuestaredneuronalo = rango[0]
  mensajeprediccion =  "No se puedo determinar la respuesta de la prediccion"
  if (respuestaredneuronalo == 1):
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia bajo"
  elif (respuestaredneuronalo == 2): 
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia medio"
  elif (respuestaredneuronalo == 3): 
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia Alto"
  
  #Mostrar puntaje de precisión de la predicción 
  score = mi_red_neuronal.score(X_test, y_test)
  
  #imprimir resultados
  print(mi_red_neuronal.score(X_test, y_test))
  print(mensajeprediccion)
  
  #Convertir puntaje en porcentaje
  porcentaje = score * 100 
  str.format("{:.2f}%".format(porcentaje))
  resultado =  str.format("{:.2f}%".format(porcentaje))

  variables = ['iescuela', 'trabajas', 'vision','gastotra','tiempoce', 'becaestu', 'califrec', 'pisocasa','aguaserv','luzelect','internet', 'vivencas', 'trabajop','cvivestu','jefecasa', 'jefeesco','saludafi','estratos']

  for var in variables:
    plt.figure(figsize=(10,8)) # Creating a rectangle (figure) for each plot
    # Regression Plot also by default includes
    # best-fitting regression line
    # which can be turned off via `fit_reg=False`
    sns.regplot(x=var, y='rangoina', data=dataset).set(title=f'Regression plot of {var} and inasistencia');
  correlations = dataset.corr()
  #annot=True #displays the correlation values
  sns.heatmap(correlations, annot=True, fmt='.1f', square=True).set(title='Heatmap of Ton Data - Pearson Correlations');
  
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
  
  print("Esta es la lista")
  print(lista)
  print("esta es la lista con los campos")
  print(listacampo)

  
  listaordenada = sorted(listacampo, key=lambda x: x[1], reverse=True)

  #imprimir lista ordenada
  print("Esta es la lista ordenada")
  print(listaordenada)


  maximo = max(lista)
  print("indice")

  campomayor = obtener_nombre_campo(listacampo, maximo)
  indice_maximo_valor = obtener_indice_tupla(listacampo, campomayor)
  print("El valor a modificar en el caso nuevo es")
  print(newcase[indice_maximo_valor])
  print("Este es el indice maximo valor")
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

  new = newcase[indice_maximo_valor]

  campo_mod = obtener_nombre_campo(listacampo, maximo)
  print("indice de campo mod")
  print(campo_mod[indice_maximo_valor])

  column_list = corr_matrix.columns.tolist()

  print(column_list)
  
  valororiginal = newcase[indice_maximo_valor]
  print("modificacion del valor original")
  print(valororiginal)
  nuevovalor = None 

  respuestaredneuronals = None 
  campos = None
  campo = listaordenada[0]
  
  b = True 
  if respuestaredneuronalo <= 1:
    print("No es necesario la recomendación, ya que el estudiante puede tener un rango de inasistencia bajo")
  else:

      # Ciclo para ajustar el valor del campo mientras respuestaredneuronals > respuestaredneuronalo
    for campos in listaordenada:
      valororiginal = newcase[indice_maximo_valor]
      while valororiginal > 0:
        valororiginal = valororiginal - 1
        print("modificamos del valor original")
        print(valororiginal)
        #Reconfigurar el caso con el nuevo valor
        newcase[indice_maximo_valor] = valororiginal
        #Hacer la simulación de la predicción con la red neuronal
        simulacion = mi_red_neuronal.predict([newcase])
        respuestaredneuronals = simulacion[0]
        #Mostrar el resultado de la simulación
        print("Estos son los datos ajustados con la inteligencia artificial de la simulación:")
        print(simulacion)
        if respuestaredneuronals == 1:
          print("El estudiante puede tener un rango de inasistencia bajo.")
        elif respuestaredneuronals == 2:
          print("El estudiante puede tener un rango de inasistencia medio.")
        elif respuestaredneuronals == 3:
          print("El estudiante puede tener  un rango de inasistencia alto.")
        #Determinar si se debe seguir ajustando el valor del campo
        if respuestaredneuronals < respuestaredneuronalo:
          print(f"La opción '{valororiginal}' del campo '{campos[0]}' es la que cambia el resultado de '{respuestaredneuronalo}' a '{respuestaredneuronals}'.")
          b = True
          break
      # Si llegamos al final del while y no se cumple la condición, el ciclo for continuará con el siguiente campo de la lista
      
          # Actualizar los valores para la próxima iteración
        maximo = max(lista)
        indice_maximo_valor = obtener_indice_tupla(listacampo, campomayor)
        valororiginal = newcase[indice_maximo_valor]
      if b:
        print("Proceso terminado.") 
        break
  print("Recorrido completamente")
  
  print(campos[0])
  print(valororiginal)
  
  return render_template('showprediction.html', variable = mensajeprediccion, score = resultado)  


  
 

 
