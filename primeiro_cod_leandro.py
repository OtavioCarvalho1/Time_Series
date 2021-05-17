
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import statistics
from sklearn.preprocessing import StandardScaler
from math import sqrt
import statsmodels.tsa as stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import acf, pacf, graphics
from time import time

### Importação dos experimentos ###

df1 = genfromtxt('PL0402_Ref2_1_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df2 = genfromtxt('PL0402_Ref2_2_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df3 = genfromtxt('PL0402_Ref2_3_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df4 = genfromtxt('PL0402_Ref2_4_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df5 = genfromtxt('PL0402_Ref2_5_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df6 = genfromtxt('PL0402_Ref2_6_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df7 = genfromtxt('PL0402_Ref2_7_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df8 = genfromtxt('PL0402_Ref2_8_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df9 = genfromtxt('PL0402_Ref2_9_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df10 = genfromtxt('PL0402_Ref2_10_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df11 = genfromtxt('PL0402_Ref3_1_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df12 = genfromtxt('PL0402_Ref3_2_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df13 = genfromtxt('PL0402_Ref3_3_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df14 = genfromtxt('PL0402_Ref3_4_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df15 = genfromtxt('PL0402_Ref3_5_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df16 = genfromtxt('PL0402_Ref3_6_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df17 = genfromtxt('PL0402_Ref3_7_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df18 = genfromtxt('PL0402_Ref3_8_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df19 = genfromtxt('PL0402_Ref3_9_150KHz__5ciclos_B3_A3.csv', delimiter=',')
df20 = genfromtxt('PL0402_Ref3_10_150KHz__5ciclos_B3_A3.csv', delimiter=',')

##### Média dos valores #####

#df_mean = (df1 + df2 + df3 + df4 + df5 + df6 + df7 + df8 + df9 + df10 + df11 + df12 + df13 + df14 + df15 + df16 + df17 + df18 + df19 + df20)/20

# print('resultado da divisão:')
# print(df_mean)
# print()

##### Nomenclatura das colunas dos dataset df1 e df2 #####

headers = ['tempo [s]', 'Amplitude PZT Atuador', 'Amplitude PZT Sensor']

df1_train = pd.DataFrame(df1, columns=headers)
# df1_tempo_sensor = df1[['tempo [s]', 'Amplitude PZT Sensor']]
df2_test = pd.DataFrame(df2, columns=headers)
# df2_tempo_sensor = df2[['tempo [s]', 'Amplitude PZT Sensor']]

##### Plot do Tempo VS Aplitude do PZT Sensor #####

# #df1
# sinal_df1_tempo = df1[['tempo [s]']]
# sinal_df1_sensor = df1[['Amplitude PZT Sensor']]
# #df2
# sinal_df2_tempo = df2[['tempo [s]']]
# sinal_df2_sensor = df2[['Amplitude PZT Sensor']]

# Plotando a Amplitude de sinal do PZT sensor
plt.figure(figsize=(14,5))
plt.plot(df1_train['tempo [s]'], df1_train['Amplitude PZT Sensor'])
plt.title("Sinal VS Tempo - df1_train")
plt.show()
plt.figure(figsize=(14,5))
plt.plot(df2_test['tempo [s]'], df2_test['Amplitude PZT Sensor'])
plt.title("Sinal VS Tempo - df2_test")
plt.show()

#%%## Standardization #####

""" Plotando o histograma de valores, para ver se o formato é gaussiano e
 dessa forma verificar se é possível utilizar o StandardScaler """

df1_train[['Amplitude PZT Sensor']].hist()
plt.title("Histograma df1")
plt.show()
df1_train[['Amplitude PZT Sensor']].hist()
plt.title("Histograma df2")
plt.show()

scaler = StandardScaler()

### df1_train standardization
scaler = scaler.fit(df1_train[['Amplitude PZT Sensor']])
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
df1_train_std = scaler.transform(df1_train[['Amplitude PZT Sensor']])
print(df1_train_std)

plt.figure(figsize=(14,5))
plt.plot(df1_train['tempo [s]'], df1_train_std)
plt.title("Standardized - Sinal VS Tempo - df1_train")
plt.show()

df1_train_std = pd.DataFrame(df1_train_std, columns=["Sensor"]) # convertendo para um Dataframe do Pandas e nomeando a coluna dele de "Sensor"
print("dataframe após padronização: ", df1_train_std)
df1_train_std.hist(histtype='barstacked')
plt.title("Histograma df1")
plt.show()

# Adicionando a coluna tempo ao dataset Padronizado
col_tempo = df1_train["tempo [s]"]
df1_train_std = df1_train_std.join(col_tempo)
df1_train_std = df1_train_std[["tempo [s]", "Sensor"]]
print(df1_train_std)


### df2_test standardization
scaler = scaler.fit(df2_test[['Amplitude PZT Sensor']])
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))
df2_test_std = scaler.transform(df2_test[['Amplitude PZT Sensor']])
print(df2_test_std)

plt.figure(figsize=(14,5))
plt.plot(df2_test['tempo [s]'], df2_test_std)
plt.title("Standardized - Sinal VS Tempo - df2_test")
plt.show()

df2_test_std = pd.DataFrame(df2_test_std, columns=["Sensor"]) # convertendo para um Dataframe do Pandas e nomeando a coluna dele de "Sensor"
print("dataframe após padronização: ", df2_test_std)
df2_test_std.hist(histtype='barstacked')
plt.title("Histograma df1")
plt.show()

# Adicionando a coluna tempo ao dataset Padronizado
col_tempo = df2_test["tempo [s]"]
df2_test_std = df2_test_std.join(col_tempo)
df2_test_std = df2_test_std[["tempo [s]", "Sensor"]]
print(df2_test_std)


#%%## Subframe entre 200 e 230 us #####

### para df1_train
df1_A0 = df1_train_std.iloc[2000:2300] 
# tentar: housing.loc[1:7,['population', 'households']] # para selecionar linhas e colunas específicas
print(df1_A0)
df1_A0.plot(x="tempo [s]", y=["Sensor"])
print()
# Mude o índice do subset para ao invés de começar em 2000, começar em 1. O statsmodel é sensível a número do índice.
df1_A0.reset_index(inplace=True, drop=True) # Serve para zerar o índice do dataset (só funcionou depois que usei o "inplace"
print(df1_A0)

### para df2_test
df2_A0 = df2_test_std.iloc[2000:2300] 
# tentar: housing.loc[1:7,['population', 'households']] # para selecionar linhas e colunas específicas
print(df2_A0)
df2_A0.plot(x="tempo [s]", y=["Sensor"])
print()
# Mude o índice do subset para ao invés de começar em 2000, começar em 1. O statsmodel é sensível a número do índice.
df1_A0.reset_index(inplace=True, drop=True) # Serve para zerar o índice do dataset (só funcionou depois que usei o "inplace"
print(df2_A0)

#%%### Checando a aleatoriedade dos dados #####

# Lag plot
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot

plt.figure(figsize=(14,5))
lag_plot(df1_train_std["Sensor"])
plt.title("Lag plot")

# Autocorrelation plot
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(14,5))

autocorrelation_plot(df1_train_std["Sensor"])
plt.title("Autocorrelation plot")

# Autocorrelation Function (ACF) e Pacial Correlation Funcion (PACF)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df1_A0["Sensor"])

plot_pacf(df1_A0["Sensor"], lags=30)
plt.title("PACF do subframe 200 a 230 us (Padronizado)")
plot_pacf(df1_train_std["Sensor"], lags=30)
plt.title("PACF do sinal padronizado")
plot_pacf(df1_train["Amplitude PZT Sensor"], lags=30)
plt.title("PACF do sinal bruto")
plt.figure()

#%%## Treino e teste do AR #####

# Treino
model = AutoReg(df1_A0["Sensor"], lags=2, old_names=False).fit()
print(model.summary())
model.plot_predict()
df1_A0.plot()

fig = plt.figure(figsize=(16,9))
fig = model.plot_diagnostics(fig=fig, lags=30)
plt.show()

pred = model.get_prediction(dynamic=False)
pred_ci = pred.conf_int()
ax = df1_A0["Sensor"].plot(label='Dado original')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Índice')
ax.set_ylabel('Amplitude')
plt.legend()
plt.show()
