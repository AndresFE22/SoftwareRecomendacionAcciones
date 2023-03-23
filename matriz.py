import numpy as np
import pandas as pd

# Carga la matriz de transición desde el archivo Excel
matriz_transicion = pd.read_excel('MATRIZL.xlsx', index_col=0)

print(matriz_transicion.iloc[:3, :3])

