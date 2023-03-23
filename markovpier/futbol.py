import numpy as np
import random as rm
import pandas as pd 


# The statespace
states = ["Edwin_Velasco","Contrario","Didier_Moreno","Walmer_Pacheco","Jose_Ortiz",
            "Edwin_Cetre","Carlos_Bacca","Gol","Yesus_Cabrera","Nelson_Deossa","Jhon_Pajoy",
            "Luis_Cariaco","Jorge_Arias","Daniel_Giraldo","Cesar_Haydar","Fabian_Sambueza",
            "Remate_al_arco","Fin_Jugada"]



# Probabilities matrix (transition matrix)
transitionMatrix = [[0.00,1.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00],
                    [0.00,0.18,0.00,0.09,0.00,0.00,0.09,0.09,0.09,0.00,0.27,0.09,0.00,0.10,0.00,0.00,0.00,0.00],
                    [0.00,0.20,0.00,0.00,0.20,0.10,0.10,0.00,0.10,0.00,0.10,0.00,0.00,0.10,0.00,0.10,0.00,0.00],
                    [0.00,0.50,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.17,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.50,0.00,0.00,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.25,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.16,0.00,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.17,0.17,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00],
                    [0.11,0.33,0.11,0.00,0.00,0.00,0.00,0.00,0.00,0.11,0.22,0.12,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.33,0.00,0.33,0.00,0.00,0.34,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.30,0.00,0.10,0.00,0.00,0.10,0.00,0.20,0.00,0.00,0.00,0.00,0.10,0.00,0.00,0.20,0.00],
                    [0.00,0.40,0.20,0.20,0.00,0.00,0.00,0.00,0.20,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.00,0.50,0.00,0.25,0.00,0.25,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.67,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.33,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0.00,0.00,0.00],
                    [0.00,0.00,0.50,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.50,0.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00],
                    [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.00]]


if sum(transitionMatrix[0])+sum(transitionMatrix[1])+sum(transitionMatrix[2])+sum(transitionMatrix[3])+sum(transitionMatrix[4])+sum(transitionMatrix[5])+sum(transitionMatrix[6])+sum(transitionMatrix[7])+sum(transitionMatrix[8])+sum(transitionMatrix[9])+sum(transitionMatrix[10])+sum(transitionMatrix[11])+sum(transitionMatrix[12])+sum(transitionMatrix[13])+sum(transitionMatrix[14])+sum(transitionMatrix[15])+sum(transitionMatrix[16])+sum(transitionMatrix[17]) != 18:
    print("En algún lugar, algo salió mal. ¿Matriz de transición, tal vez?")
else: print("¡¡Todo va a estar bien, deberías seguir adelante!! ;)")

# Imprimir matriz de probabilidades (matriz de transición)
pd.DataFrame(transitionMatrix,columns=states,index=states)

def nextStateProbability(current_state, next_state):



    #index estado actual
    ics = states.index(current_state)
  #index siguiente estado
    ins = states.index(next_state)
    transition_probability = transitionMatrix[ics][ins]
    print("Probability from state [{}] to  State [{}] is {}".format(current_state, next_state, transition_probability))
    print("La probabilidad del siguiente estado----> " )

    return transition_probability
  #print ('Index of %s is %s', current_state, ics)
  #list.index(x[, start[, end]])

input_current_state = "Edwin_Velasco"
input_next_state = "Edwin_Cetre"

print(nextStateProbability(input_current_state, input_next_state))
