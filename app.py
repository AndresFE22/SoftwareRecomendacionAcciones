from flask import Flask, render_template, request, session, jsonify
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
  
  
@app.route('/actualizarDS')
def newdt():
  archivo = request.files['archivo_csv']
  archivo.save('datasetPrediccionprueba.csv')
  
  #Actualizar Dataset
  global dataset 
  dataset = pd.read_csv('datasetPrediccionprueba.csv', delimiter=';')
  
  
  return jsonify('msg')



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
  if (respuestaredneuronalo == 0):
    mensajeprediccion = "Bajo"
  elif (respuestaredneuronalo == 1): 
    mensajeprediccion = "Medio"
  elif (respuestaredneuronalo == 2): 
    mensajeprediccion = "Alto"
  
  #Mostrar puntaje de precisión de la predicción 
  score = mi_red_neuronal.score(X_test, y_test)
  
  #imprimir resultados
  print(mi_red_neuronal.score(X_test, y_test))
  print(mensajeprediccion)
  
  #Convertir puntaje en porcentaje
  porcentaje = score * 100 
  str.format("{:.2f}%".format(porcentaje))
  resultado =  str.format("{:.2f}%".format(porcentaje))

  variables = ['iescuela', 'trabajas', 'vision', 'gastotra', 'tiempoce', 'becaestu', 'califrec', 'pisocasa', 'aguaserv', 'luzelect', 'internet', 'vivencas', 'trabajop', 'vivescon', 'parenjef', 'escojefe', 'saludafi', 'estratos']

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

  campos_ordenados = [tupla[0] for tupla in listaordenada]
  print(f"Campos ordenados {campos_ordenados}")
  
  
  
  
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

  
  
  campo_mod = obtener_nombre_campo(listacampo, maximo)
  print("indice de campo mod")
  print(campo_mod[indice_maximo_valor])

  column_list = corr_matrix.columns.tolist()

  print(column_list)
  
  valororiginal = newcase[indice_maximo_valor]
  print("modificacion del valor original")
  print(valororiginal)
  
  nueva_lista = newcase
  print(f"Esta es la nueva lista: {nueva_lista}")
  
  respuestaredneuronals = None 
  global campos 
  campos = None
  b = False
  
  asociacion = {}
  
  for campo, valor in zip(variables, newcase):
    if campo not in asociacion:
        asociacion[campo] = str(valor)

  print(f"asociacion: {asociacion}")

  newcase_ordenado = []

  # Ordenar newcase según campos_ordenados
  for campo in campos_ordenados:
    print(f"Voy por el campo {campo}")
    # Buscar el valor correspondiente al campo en la asociación
    valor = asociacion.get(campo)
    print(f"Valor es {valor}")
    # Verificar si el valor existe en la lista newcase
    if valor is not None:
        # Agregar el valor correspondiente a newcase_ordenado
        newcase_ordenado.append(int(valor))

  print("newcase ordenado:", newcase_ordenado)

  if respuestaredneuronalo == 0:
      print("No es necesario la recomendación, ya que el estudiante puede tener un rango de inasistencia bajo")
  else:
    # Agregar mensaje de depuración
    print(f"Lista ordenada: {listaordenada}")
    
    # Ciclo para ajustar el valor del campo mientras respuestaredneuronals > respuestaredneuronal
    for campos in campos_ordenados:
      respuestaredneuronals = None 
      indice_campo = campos_ordenados.index(campos)
      print(f"Este es el indice {indice_campo}")

      valororiginal = newcase_ordenado[indice_campo]
      #Agregar mensaje de depuración
      print(f"Campos: {campos}, valororiginal: {valororiginal}")
      
      while valororiginal > 0:
        valororiginal = valororiginal - 1
        # Agregar mensaje de depuración
        print("modificamos del valor original")
        print(valororiginal)
        
        newcase_ordenado[indice_campo] = valororiginal
        
        #Hacer la simulación de la predicción con la red neuronal
        simulacion = mi_red_neuronal.predict([newcase_ordenado])
        respuestaredneuronals = simulacion[0]
        
        #Mostrar el resultado de la simulación
        print("Estos son los datos ajustados con la inteligencia artificial de la simulación:")
        print(simulacion)
        
        if respuestaredneuronals == 0:
          print("El estudiante puede tener un rango de inasistencia bajo.")
        elif respuestaredneuronals == 1:
          print("El estudiante puede tener un rango de inasistencia medio.")
        elif respuestaredneuronals == 2:
          print("El estudiante puede tener  un rango de inasistencia alto.")
            
        #Determinar si se debe seguir ajustando el valor del campo
        if respuestaredneuronals < respuestaredneuronalo:
          print(f"La opción '{valororiginal}' del campo '{campos}' es la que cambia el resultado de '{respuestaredneuronalo}' a '{respuestaredneuronals}'.")
          b = True
          break
        
    
      valororiginal = newcase_ordenado[indice_campo]
      
      #Agregar mensaje de depuración
      print(f"Valor original: {valororiginal}, b: {b}")

      # Ordenar newcase según campos_ordenados
      if b:
        print("Proceso terminado.")
        break 


  if campos is not None:
    print(campos[0])
  else:
    print("Campos es none")
    print(valororiginal)
    
    #Elección de la recomendación
    
      #Definir los códigos
  codigos = {
    'iescuela' : { 
      'Por irme de viaje': 1, 'Cambio de domicilio': 2, 'Por el trabajo': 3, 'Problemas familiares': 4, 'Problemas de salud': 5, 'Me expulsaron/suspendieron': 6},

    'trabajas' : {
      'Si': 1, 'No':2},

    'vision' : {
      'Seguir estudiando y buscar un trabajo': 1, 'Seguir estudiando y continuar trabajando': 2, 'Seguir estudiando y dejar de trabajar': 3, 'Sólo seguir estudiando': 4, 'Dejar de estudiar y dedicarme a trabajar': 5},
      
    'gastotra' : {
      'Menos o igual a $6000': 1, 'Más de $6000': 2},
      
    'tiempoce' : {
      'Menos o igual a 10 min': 1, 'Entre 11 a 30 min': 2, 'Mas de 30 min': 3},

    'becaestu' : {
      'Si': 1, 'No':2},

    'califrec' : { 
      'Altas': 1, 'Medias': 2, 'Bajas': 3},

    'pisocasa' : {
      'Cemento': 1, 'Mosaico, madera u otro recubrimiento': 2, 'Tierra': 3},

    'aguaserv' : {
      'Si': 1, 'No':2},
      
    'luzelect' : {
      'Si': 1, 'No':2},
      
    'internet' : {
     'Si': 1, 'No':2},

    'vivencas' : {
      'Si': 1, 'No':2},

    'iescuela' : { 
      'Menos o igual a 4': 1, 'Entre 5 a 10': 2, 'Mas de 10': 3},

    'trabajop' : {
      'Si': 1, 'No':2},

    'vivescon' : {
      'Padre': 1, 'Madre': 2, 'Hermanos(as)': 3, 'Otros familiares': 4, 'Otras personas que no son familiares': 5, 'Solo(a)': 6, 'Con amigos': 7},
      
    'escojefe' : {
      'Posgrado': 1, 'Profesional': 2, 'Técnico': 3, 'Preparatoria completa': 4, 'Preparatoria incompleta': 5, 'Secundaria completa': 6, 'Secundaria incompleta': 7, 'Primaria completa': 8, 'Primaria incompleta': 9, 'Sin escolaridad': 10, 'No tengo papá': 11, 'No lo sé': 12},
      
    'parenjef' : {
      'Yo soy el hijo': 1, 'Yo soy el(la) cónyuge': 2, 'Yo soy un familiar': 3, 'Yo no soy familiar': 4, 'Yo soy el jefe del hogar': 5, 'Otro': 6},

    'saludafi' : {
      'Si': 1, 'No':2},

    'estratos' : {
       '1(A)': 1, '2(B)': 2, '3(C)': 3, '4(D)': 4
    }
       
    }
  

  print(f"Estas son las variables {codigos}, {campos}, {valororiginal}")
    
   # Guardar las variables en la sesión
  session['codigos'] = codigos
  session['campos'] = campos
  session['valororiginal'] = valororiginal
    
    
  return render_template('showprediction.html', variable = mensajeprediccion , score = resultado)  


@app.route('/recomendation', methods = ['GET', 'POST'])
def recom():

  codigos = session.get('codigos')
  campos = session.get('campos')
  valororiginal = session.get('valororiginal')

  # Definir la función para calcular la distancia de Hamming
  def hamming_distance(str1, str2):
    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance

  def get_recommendation_by_field(row, field, recommendation_col):
    value = row[field]
    recommendations = df.loc[df[field] == value, recommendation_col].tolist()
    if recommendations:
        distances = [hamming_distance(value.lower(), rec.lower()) for rec in recommendations]
        min_dist = min(distances)
        return recommendations[distances.index(min_dist)]
    return None
  
  print("Campo:", df[campos])
  #Definir valor
  valor_codificado = valororiginal
  valor = list(codigos[campos].keys())[list(codigos[campos].values()).index(valor_codificado)]
  print("Este esa la opcion", valor)
  recomendaciones_col = f"{campos}r"
  recommendation = None

  # Encontrar las filas que corresponden al valor específico del campo
  mask = (df[campos] == valor)
  print(mask)

  # Buscar la recomendación en la columna correspondiente
  for index, row in df.iterrows():
    if mask[index]:
        recommendation = get_recommendation_by_field(row, campos, recomendaciones_col)
        if recommendation:
            break
        
  if recommendation:
    print(f"La recomendación para el campo {campos} y el valor {valor} es: {recommendation}")
  else:
    print(f"No se encontraron recomendaciones para el campo {campos} y el valor {valor}")

  recomendaciones = recommendation.split(';')

  print("Recomendaciones escogidas")
  print(recomendaciones)

  dff = dfh[dfh['Recom'].isin(recomendaciones)]

  print("Heurística:", dff)

  
  return render_template('recomendaciones.html', dff=dff)
  



