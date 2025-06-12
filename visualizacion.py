import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Crear el DataFrame manualmente (N/A como np.nan)
data = {
    'Alternativa': ['GPT-4o', 'GPT-4 Turbo', 'Gemini Ultra 1.0', 'Llama 3', 'Claude 3 Opus', 'Gemini 1.5'],
    'C5': [88.7, 86.5, 83.7, 86.1, 86.8, 81.9],
    'C6': [53.6, 48, np.nan, 48, 50.4, np.nan],
    'C7': [76.6, 72.6, 53.2, 57.8, 60.1, 58.5],
    'C8': [90.2, 87.1, 74.4, 84.1, 84.9, 71.9],
    'C9': [83.4, 86, 82.4, 83.5, 83.1, 78.9],
}

df = pd.DataFrame(data)

# Guardar una copia sin la columna 'Alternativa'
df_numerico = df.drop(columns=['Alternativa'])

# 1. Matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(df_numerico.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de correlación entre criterios')
plt.show()

# 2. Diagramas de dispersión con líneas de regresión (solo variables completas)
sns.pairplot(df_numerico, kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.suptitle('Relaciones entre criterios (Regresión Lineal)', y=1.02)
plt.show()

