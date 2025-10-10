import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import huffmancodec as huffc

#---------------------------------------------------------------
#---------------------------------------------------------------
# Alínea 1
#---------------------------------------------------------------
#---------------------------------------------------------------

df = pd.read_excel("Data/CarDataset.xlsx")

# Matriz de valores 
X = df.astype(np.uint16).astype(np.uint16).values

# Lista com os nomes das variáveis
variaveis = df.columns.tolist()

print("Dimensão da matriz:", X.shape)
print("Nomes das variáveis:", variaveis)

#---------------------------------------------------------------
#---------------------------------------------------------------
# Alínea 2
#---------------------------------------------------------------
#---------------------------------------------------------------

# Lista de variáveis independentes (todas exceto MPG)
variaveis = [col for col in df.columns if col != "MPG"]

# Criar figura e eixos (2 linhas, 3 colunas)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Ajustar espaçamento
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Iterar sobre variáveis e eixos
for ax, var in zip(axes.ravel(), variaveis):
    ax.scatter(df[var], df["MPG"], alpha=0.6)
    ax.set_xlabel(var)
    ax.set_ylabel("MPG")
    ax.set_title(f"MPG vs {var}")
    ax.grid(True, linestyle="--", alpha=0.5)

plt.show()

#---------------------------------------------------------------
#---------------------------------------------------------------
# Alínea 3
#---------------------------------------------------------------
#---------------------------------------------------------------

alfabetos = {}

for col in df.columns:
    simbolos = df[col].astype(np.uint16).unique()
    alfabetos[col] = sorted(simbolos.tolist())
    print(f"Variável: {col}")
    print(f"  Tamanho do alfabeto: {len(alfabetos[col])}")
    print(f"  Exemplos: {alfabetos[col][:10]}{'...' if len(alfabetos[col]) > 10 else ''}\n")

#---------------------------------------------------------------
#---------------------------------------------------------------
# Alínea 4
#---------------------------------------------------------------
#---------------------------------------------------------------

def calcular_ocorrencias(df):
    """
    Calcula o número de ocorrências de cada símbolo
    para cada variável do dataframe.
    
    Retorna um dicionário:
    {variável: {símbolo: contagem}}
    """
    ocorrencias = {}
    
    for col in df.columns:
        valores = df[col].values
        unicos, contagens = np.unique(valores, return_counts=True)
        ocorrencias[col] = dict(zip(unicos, contagens))
    
    return ocorrencias

ocorrencias = calcular_ocorrencias(df)

# Exemplo para a variável Cylinders
for simbolo, freq in ocorrencias["Cylinders"].items():
    print(f"Símbolo {simbolo}: {freq} ocorrências")

#---------------------------------------------------------------
#---------------------------------------------------------------
# Alínea 5
#---------------------------------------------------------------
#---------------------------------------------------------------

def plot_ocorrencias(ocorrencias):
    """
    Recebe um dicionário {variável: {símbolo: contagem}}
    e gera gráficos de barras para cada variável.
    """
    for var, freq_dict in ocorrencias.items():
        # Filtrar apenas símbolos com ocorrências > 0
        simbolos = list(freq_dict.keys())
        contagens = list(freq_dict.values())
        
        plt.figure(figsize=(8,5))
        plt.bar(simbolos, contagens, color="skyblue", edgecolor="black")
        plt.title(f"Distribuição de símbolos - {var}")
        plt.xlabel("Símbolos do alfabeto")
        plt.ylabel("Número de ocorrências")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.show()

plot_ocorrencias(ocorrencias)
