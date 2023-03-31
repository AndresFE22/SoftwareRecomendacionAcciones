from flask import Flask, render_template, request
import numpy as np
import random as rm
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.utils import check_array
from sklearn.ensemble  import RandomForestClassifier 
from sklearn.neural_network import MLPClassifier

#import matplotlib 
#import matplotlib.pyplot as plt 
#import seaborn as sns 
dataset = pd.read_csv('datasetPrediccionprueba.csv', delimiter= ";") 
app = Flask(__name__) 

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
 
 

  return render_template('training.html')

@app.route('/showpageprediction')
def showpageprediction():
  
  return render_template('formprediction.html', variable= "alto")
  


@app.route('/doprediction', methods=['POST'])
def doprediction():



#global showPrediction 
  iescuela = int(request.form['iescuela'])
  #print ("escuela", iescuela)
  trabajar = int(request.form['trabajar'])
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


  X = dataset.iloc[:, 0:18].values
  y = dataset.iloc[:, -1].values



#dividir los datos en etapas de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Algoritmo de red neuronal
  iactivation = "logistic"
  isolver = "lbfgs"
  imaxiter = 1000

  mi_red_neuronal= MLPClassifier(activation=iactivation,
                                random_state=1,
                                solver= isolver,
                                max_iter=imaxiter).fit(X_train, y_train)
#Predicción 

  rango = mi_red_neuronal.predict([[iescuela,trabajar,vision,gastotra,tiempoce,becaestu,califrec,pisocasa,aguaserv,luzelect,internet,vivencas,trabajop,vivescon,parenjef,escojefe,saludafi,estratos]])
#Nueva prediccion
  newcase = mi_red_neuronal.predict([[iescuela,trabajar,vision,gastotra,tiempoce,becaestu,califrec,pisocasa,aguaserv,luzelect,internet,vivencas,trabajop,vivescon,parenjef,escojefe,saludafi,estratos]])
  

  #prediccion=showPrediction(rango[0])
  y_pred = mi_red_neuronal.predict(X_test)
  respuestaredneuronal = rango[0]
  mensajeprediccion =  "No se puedo determinar la respuesta de la prediccion"
  if (respuestaredneuronal == 0):
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia bajo"
  elif (respuestaredneuronal == 1): 
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia medio"
  elif (respuestaredneuronal == 2): 
    mensajeprediccion = "El estudiante puede tener un rango de inasistencia Alto"
  
  #Mostrar puntaje de precisión de la predicción 
  score = mi_red_neuronal.score(X_test, y_test)
  
  #Convertir puntaje en porcentaje
  porcentaje = score * 100 
  str.format("{:.2f}%".format(porcentaje))
  resultado =  str.format("{:.2f}%".format(porcentaje))

  
  
 



  
  
  #check_array(rango, dtype='numeric')
 
 

  #Algoritmo de random forest
  
  #rfc=RandomForestClassifier(n_estimators=200)
  #rfc.fit(X_train,y_train) 
  #rango = rfc.predict([[iescuela, trabajas, vision, gastotra, tiempoce, becaestu, califrec, pisocasa, aguaserv, luzelect, internet, vivencas, trabajop, vivescon, escojefe, parenjef, saludafi, estratos]])


  #def showPrediction(argument): 

    #switcher = {
      #1: "rf: Rango de inasistencia Alto",
      #2: "rf: Rango de inasistencia Medio",
      #3: "rf: Rango de inasistencia Bajo"
    #}
  #showPrediction(rango[0])

 
  #return render_template('showprediction.html', variable= showPrediction(rango[0]))  
  #return render_template('showprediction.html', variable= str(rango[0]))  
  return render_template('showprediction.html', variable = mensajeprediccion, score = resultado)  


  
 

 
