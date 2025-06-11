import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Crear DataFrame con valores actualizados (coma convertida a punto)
data = {
    'Alternativa': ['GPT-4o', 'GPT-4 Turbo', 'Gemini Ultra 1.0', 'Llama 3', 'Claude 3 Opus', 'Gemini 1.5'],
    'C1': [8.8, 8.8, 8.2, 7.0, 9.0, 8.0],
    'C2': [10, 3, 3, 10, 3, 5],
    'C3': [8, 9, 7, 6, 7, 7],
    'C4': [8.8, 8.8, 8.6, 9.0, 8.8, 8.6],
    'C5': [88.7, 86.5, 83.7, 86.1, 86.8, 81.9],
    'C6': [53.6, 48.0, np.nan, 48.0, 50.4, np.nan],
    'C7': [76.6, 72.6, 53.2, 57.8, 60.1, 58.5],
    'C8': [90.2, 87.1, 74.4, 84.1, 84.9, 71.9],
    'C9': [83.4, 86.0, 82.4, 83.5, 83.1, 78.9]
}

df = pd.DataFrame(data)

# Separar datos completos e incompletos para C6
df_complete = df[df['C6'].notna()]
df_missing = df[df['C6'].isna()]

# Variables predictoras
X_train = df_complete.drop(columns=['Alternativa', 'C6'])
y_train = df_complete['C6']
X_pred = df_missing.drop(columns=['Alternativa', 'C6'])

# Modelo de regresión
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir valores faltantes
predicted_c6 = model.predict(X_pred)

# Reemplazar valores faltantes e imprimir mensajes
for i, idx in enumerate(df_missing.index):
    pred_value = predicted_c6[i]
    alternativa = df.loc[idx, 'Alternativa']
    df.loc[idx, 'C6'] = pred_value
    print(f"Valor faltante del C6 para {alternativa} = {pred_value:.2f}")

# Mostrar tabla completa
print("\nTabla completa con C6 imputado:")
print(df.round(2))