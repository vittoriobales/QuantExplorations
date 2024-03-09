# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:23:32 2024

@author: vitto
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


####### DATI PUN #################

# Dati forniti
data = pd.read_excel() #insert PATH dati PUN

# Creare un DataFrame
df = pd.DataFrame(data)
df2= pd.DataFrame()

df2 = pd.concat([df2, df.iloc[:, 1]], axis=1)

date_range = pd.date_range(start='2019-01-01', end='2023-12-31', freq='D')
data_PUN = df2.set_index(date_range)

# Visualizzare il DataFrame risultante
print(df)

data_oilvix = pd.read_excel() #insert PATH dati OIL VIX
data_oilvix=data_oilvix.dropna()

data_emerg_markt= pd.read_excel() #insert PATH dati Emerg. markets VOL INDEX      
data_emerg_markt= data_emerg_markt.dropna()



#Dati generation 

generation_data = pd.read_excel() #insert PATH dati Actual Generation(TERNA)
# Creiamo una colonna 'Mese' basata sulla colonna 'Data'
generation_data['Date'] = pd.to_datetime(generation_data['Date'])
generation_data['giorno'] = generation_data['Date'].dt.to_period('D')

# Raggruppiamo per 'Mese' e 'Tipologia' e sommiamo i valori
dati_mensili_generation = generation_data.groupby(['giorno', 'Primary Source'])['Actual Generation [GWh]'].sum().reset_index()

# Visualizziamo il DataFrame risultante
print(dati_mensili_generation)

# Utilizziamo il metodo pivot per trasporre i valori dalla riga alla colonna
generation_data_clean = dati_mensili_generation.pivot(index='giorno', columns='Primary Source', values='Actual Generation [GWh]')

# Visualizziamo il DataFrame risultante
print(generation_data_clean)  


#conversione degli indici a tipo datetime
data_oilvix.index = pd.to_datetime(data_oilvix.index)
data_emerg_markt.index = pd.to_datetime(data_emerg_markt.index)
generation_data_clean.index = pd.to_datetime(generation_data_clean.index.to_timestamp())
data_PUN.index = pd.to_datetime(data_PUN.index)


common_indices = set(data_oilvix.index) & set(data_emerg_markt.index) & set(data_PUN.index) & set(generation_data_clean.index)
 
data_oilvix_common = data_oilvix[data_oilvix.index.isin(common_indices)]
data_emerg_markt_common = data_emerg_markt[data_emerg_markt.index.isin(common_indices)]
generation_data_clean_common = generation_data_clean[generation_data_clean.index.isin(common_indices)]
data_PUN_common = data_PUN[data_PUN.index.isin(common_indices)]

# Concatenazione lungo le colonne (axis=1)
merged_df = pd.concat([data_oilvix_common, data_emerg_markt_common, generation_data_clean_common, data_PUN_common], axis=1)

######## REGRESSIONE MULTIPLA ############

# Supponiamo che merged_df sia il tuo DataFrame con le variabili indipendenti e la variabile dipendente (PUN)
# Assicurati che tutte le colonne siano numeriche e non ci siano valori mancanti

# Dividi il DataFrame in variabili indipendenti (X) e variabile dipendente (y)
X = merged_df.drop('PUN', axis=1)
y = merged_df['PUN']

# Dividi il dataset in set di allenamento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea il modello di regressione lineare
model = LinearRegression()

# Allena il modello sul set di allenamento
model.fit(X_train, y_train)

# Fai previsioni sul set di test
y_pred = model.predict(X_test)

# Valuta le prestazioni del modello
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualizza i risultati della previsione rispetto ai valori effettivi
plt.scatter(y_test, y_pred)
plt.xlabel("Valori Effettivi")
plt.ylabel("Previsioni")
plt.title("Confronto tra Valori Effettivi e Previsioni")
plt.show()


####### Matrice di correlazione
correlation_matrix = merged_df.corr()
print(correlation_matrix)

# Visualizza una heatmap della matrice di correlazione
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice di Correlazione')
plt.show()

###### diagramma a dispersione

import seaborn as sns

sns.pairplot(merged_df, x_vars=X.columns, y_vars='PUN', height=3)
plt.suptitle('Pair Plot tra Variabili Indipendenti e PUN', y=1.02)
plt.show()


# statistiche descrittive

merged_df.hist(figsize=(12, 8), bins=20)
plt.suptitle('Distribuzione delle Variabili', y=1.02)
plt.show()


############### PCA ######################

from sklearn.decomposition import PCA

pca = PCA()
principal_components = pca.fit_transform(X)

# Visualizza la varianza spiegata cumulativa
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = explained_variance_ratio.cumsum()

plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Numero di Componenti Principali')
plt.ylabel('Varianza Spiegata Cumulativa')
plt.title('Analisi delle Componenti Principali (PCA)')
plt.show()

############### Feature Importance ################################

from sklearn.ensemble import RandomForestRegressor

# Crea un modello di Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Ottieni l'importanza delle variabili
feature_importances = rf_model.feature_importances_

# Visualizza l'importanza delle variabili
plt.bar(X.columns, feature_importances)
plt.xlabel('Variabili Indipendenti')
plt.ylabel('Importanza')
plt.title('Importanza delle Variabili nella Predizione del PUN')
plt.xticks(rotation=45)
plt.show()

################## SVR ######################
 
from sklearn.svm import SVR

# Support Vector Regression
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_y_pred = svr_model.predict(X_test)

# prestazioni del modello
mse_svr = mean_squared_error(y_test, svr_y_pred)
print(f'Mean Squared Error (SVR): {mse_svr}')

################# Multi-layer Perpeptron Regressor###############################

from sklearn.neural_network import MLPRegressor

# Crea un'istanza di MLPRegressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(48,32),  # Numero di neuroni nella hidden layer
                             activation='relu',            # Funzione di attivazione ('relu', 'logistic', 'tanh', etc.)
                             solver='adam',                # Algoritmo di ottimizzazione ('adam', 'sgd', etc.)
                             alpha=0.0001,                 # Termine di regolarizzazione L2
                             max_iter=1000,                # Numero massimo di iterazioni
                             random_state=42)              # Seed per la casualità (per riproducibilità)

# Addestra il modello con i tuoi dati di addestramento (X_train, y_train)
mlp_regressor.fit(X_train, y_train)

# Fai previsioni su nuovi dati (X_test)
y_pred_mlp = mlp_regressor.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("Mean Squared Error (MLP):", mse_mlp)
print("R-squared (MLP):", r2_mlp)

#grafico dei residui
residuals_mlp = y_test - y_pred_mlp
plt.scatter(y_pred_mlp, residuals_mlp)
plt.xlabel("Previste (MLP)")
plt.ylabel("Residui (MLP)")
plt.title("Grafico di Residui (MLP)")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()


##################### ARIMA-X ########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Supponiamo che merged_df sia il tuo DataFrame con le variabili indipendenti e la variabile dipendente (PUN)
# Assicurati che l'indice del DataFrame sia una data (datetime)

# Crea un DataFrame solo con le colonne selezionate
time_series_df = merged_df

# Dividi i dati in set di allenamento e set di test
train_size = int(len(time_series_df) * 0.8)
train, test = time_series_df[:train_size], time_series_df[train_size:]

# Plot delle serie storiche
plt.figure(figsize=(12, 6))
plt.plot(train['PUN'], label='Train')
plt.plot(test['PUN'], label='Test')
plt.title('Serie Storiche del PUN')
plt.xlabel('Data')
plt.ylabel('PUN')
plt.legend()
plt.show()

# Addestramento del modello ARIMA-X
col_da_escludere = 'PUN'
endog = train['PUN']
colvar = [col for col in train.columns if col != col_da_escludere]
exog = train[colvar]
# Crea un modello SARIMAX con esogeni
model = SARIMAX(endog, exog=exog, order=(2, 2, 2), seasonal_order=(0, 0, 0, 0), trend='c')

# Fai il fit del modello
results = model.fit()

# Fai previsioni sul set di test
start = len(train)
end = len(train) + len(test) - 1
predictions = results.predict(start=start, end=end, exog=test[colvar], dynamic=False)

# Valuta le prestazioni del modello
mse_arimax = mean_squared_error(test['PUN'], predictions)
print(f'Mean Squared Error (ARIMA-X): {mse_arimax}')

# Plot delle previsioni
plt.figure(figsize=(12, 6))
plt.plot(test['PUN'], label='Test')
plt.plot(predictions, label='Predictions')
plt.title('Confronto tra Test e Previsioni (ARIMA-X)')
plt.xlabel('Data')
plt.ylabel('PUN')
plt.legend()
plt.show()


################ XGBOOST

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Prepare the data
X_train_xg = train[colvar]
y_train_xg = train['PUN']
X_test_xg = test[colvar]
y_test_xg = test['PUN']

# Create an XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror',gamma=3.0,   
            random_state=42,max_depth=4,learning_rate=0.001,subsample=0.6,colsample_bytree=0.5,  
            colsample_bylevel=0.5, n_estimators=350)

# Fit the model to the training data
xgb_model.fit(X_train_xg, y_train_xg)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_xg)

# Evaluate the model
mse_xgb = mean_squared_error(y_test_xg, y_pred_xgb)
print(f'Mean Squared Error (XGBoost): {mse_xgb}')

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, y_test_xg, label='Actual')
plt.plot(test.index, y_pred_xgb, label='XGBoost Predicted')
plt.xlabel('Date')
plt.ylabel('PUN')
plt.title('XGBoost Forecast with Exogenous Variables')
plt.legend()
plt.show()


################################## LSTM Model ################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Assuming merged_df contains the DataFrame with datetime index and columns including 'PUN', 'Var1', 'Var2', 'Var3'
# If not, replace it with your actual DataFrame

# Extract the target variable (PUN) and exogenous variables
target_variable = 'PUN'
exogenous_variables = colvar

# Combine target variable and exogenous variables
data = merged_df[[target_variable] + exogenous_variables]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Define the number of time steps for input sequences and the number of features
n_steps = 10  # You can adjust this based on your preference
n_features = len(exogenous_variables) + 1  # Target variable + exogenous variables

# Prepare the data for LSTM
X, y = [], []
for i in range(len(scaled_data) - n_steps):
    X.append(scaled_data[i : i + n_steps, :])
    y.append(scaled_data[i + n_steps, 0])  # PUN is the first column

X, y = np.array(X), np.array(y)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=300, batch_size=16, verbose=1)

# Make predictions on the test set
y_pred_lstm = model.predict(X_test)

# Invert the scaling for predictions and actual values
y_pred_lstm_inv = scaler.inverse_transform(np.concatenate((y_pred_lstm, X_test[:, -1, 1:]), axis=1))[:, 0]
y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), X_test[:, -1, 1:]), axis=1))[:, 0]

# Evaluate the model
mse_lstm = mean_squared_error(y_test_inv, y_pred_lstm_inv)
print(f'Mean Squared Error (LSTM): {mse_lstm}')

# Plot the actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size + n_steps:], y_test_inv, label='Actual')
plt.plot(data.index[train_size + n_steps:], y_pred_lstm_inv, label='LSTM Predicted')
plt.xlabel('Date')
plt.ylabel('PUN')
plt.title('LSTM Forecast with Exogenous Variables')
plt.legend()
plt.show()