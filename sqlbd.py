import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib

def process_stock_data(ticker):
    # Usando o yfinance para obter os dados
    data = yf.download(ticker)
    data = data.drop(data.columns[[4,5]], axis=1)

    data['media_21'] = data['Close'].ewm(21).mean().shift()
    data['media_9'] = data['Close'].ewm(9).mean().shift()
    data = data.drop(data.index[:21])

    def relative_strength_idx(df, n=14):
        Close = df['Close']  # Use 'Close' em vez de 'Close'
        delta = Close.diff()
        delta = delta.iloc[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    data['RSI'] = relative_strength_idx(data).fillna(0)
    data = data[data['RSI'].notna() & (data['RSI'] != 0)]

    media_12 = pd.Series(data['Close'].ewm(span=12, min_periods=12).mean())
    media_26 = pd.Series(data['Close'].ewm(span=26, min_periods=26).mean())
    data['MACD'] = pd.Series(media_12 - media_26)
    data['MACD_signal'] = pd.Series(data.MACD.ewm(span=9, min_periods=9).mean())

    data.dropna(axis=0, how="any", inplace=True)
    data = data.reset_index()

    # Remover a coluna 'Date' que agora é uma coluna comum
    data = data.drop('Date', axis=1)

    return data

def update_users_news_description(signal):
    # Conecte-se ao banco de dados
    conn = sqlite3.connect('teste.db')
    cursor = conn.cursor()

    # Obter os registros que atendem ao critério
    cursor.execute(f'SELECT id, name FROM users WHERE poupança > "9000";')
    rows = cursor.fetchall()

    # Atualizar a coluna news_description com base no sinal e no nome do usuário
    for row in rows:
        user_id = row[0]
        user_name = row[1]
        news_description = f"Olá, {user_name}! De acordo com a inteligencia artificial, Hoje é um bom dia para {signal} de ações da Petrobras."
        cursor.execute(f'UPDATE users SET news_description = ? WHERE id = ?', (news_description, user_id))

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    update_users_news_description()

if __name__ == "__main__":
    # Executar apenas a função process_stock_data
    ticker = 'PETR4.SA'
    processed_data = process_stock_data(ticker)
    print(processed_data)



# import numpy as np 
# import pandas as pd
# from sklearn.model_selection import KFold
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor
# import matplotlib.pyplot as plt
# from xgboost import plot_importance, plot_tree


# #Leitura do datasete retirada do datetime
# data = pd.read_csv("petr4.csv")

# #retirada da coluna volume por estar imcompleta 
# data = data.drop(data.columns[[5]],axis=1)
# print(data)

# data['media_21'] = data['Close'].ewm(21).mean().shift()

# data['media_9'] = data['Close'].ewm(9).mean().shift()

# def relative_strength_idx(data, n=14):
#     Close = data['Close'] 
#     delta = Close.diff()
#     delta = delta.iloc[1:]
#     pricesUp = delta.copy()
#     pricesDown = delta.copy()
#     pricesUp[pricesUp < 0] = 0
#     pricesDown[pricesDown > 0] = 0
#     rollUp = pricesUp.rolling(n).mean()
#     rollDown = pricesDown.abs().rolling(n).mean()
#     rs = rollUp / rollDown
#     rsi = 100.0 - (100.0 / (1.0 + rs))
#     return rsi

# data['RSI'] = relative_strength_idx(data).fillna(0)
# data = data[data['RSI'].notna() & (data['RSI'] != 0)]

# import pandas as pd
# import matplotlib.pyplot as plt

# data = pd.DataFrame(data)
# data["Date"] = pd.to_datetime(data["Date"])

# # Plotar os gráficos
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# # Gráfico principal
# ax1.plot(data["Date"], data["Close"], label="Close")
# ax1.plot(data["Date"], data["media_21"], label="media_21")
# ax1.plot(data["Date"], data["media_9"], label="media_9")
# ax1.set_ylabel("Valores")
# ax1.legend()

# # Gráfico para RSI
# ax2.plot(data["Date"], data["RSI"], label="RSI", color="orange")
# ax2.set_xlabel("Data")
# ax2.set_ylabel("RSI")
# ax2.legend()

# # Ajustes de layout
# plt.tight_layout()

# # Mostrar os gráficos
# plt.show()

# media_12 = pd.Series(data['Close'].ewm(span=12, min_periods=12).mean())
# media_26 = pd.Series(data['Close'].ewm(span=26, min_periods=26).mean())
# data['MACD'] = pd.Series(media_12 - media_26)
# data['MACD_signal'] = pd.Series(data.MACD.ewm(span=9, min_periods=9).mean())



# data.describe()


# data.info()

# data_index = data['Date'].copy()

# # Remover a coluna "Date" do DataFrame
# data.drop('Date', axis=1, inplace=True)

# data.index = data_index

# data.dropna(axis=0, how="any", inplace=True)
# data

# future_data = data.iloc[-1:]
# data = data.drop(data.index[-1:])

# X = data.drop("Close", axis=1)
# y = data['Close']



# from sklearn.model_selection import train_test_split, KFold, GridSearchCV
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


# num_folds = 5
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

# parameters = {
#     'n_estimators': [ 100],
#     'learning_rate': [ 0.03],
#     'max_depth': [6],
#     'gamma': [0.02],
#     'subsample': [0.75],
#     'colsample_bynode': [0.5],
#     'random_state': [42]
# }

# base_model = XGBRegressor(
#     booster="gbtree",
#     objective='reg:squarederror'
# )


# clf = GridSearchCV(base_model, parameters, cv=kf)

# clf.fit(X, y)

# print(f'Melhores parametros: {clf.best_params_}')
# print(f'melhor validação = {clf.best_score_}')


# y_pred_train = clf.predict(X_train)
# mse_train = mean_squared_error(y_train, y_pred_train)
# print(f'mean_squared_error (Training) = {mse_train}')


# y_pred_val = clf.predict(X_val)
# mse_val = mean_squared_error(y_val, y_pred_val)
# print(f'mean_squared_error (Validation) = {mse_val}')

# future_close_pred = clf.predict(future_data.drop("Close", axis=1))
# print(f'Previsão de valor do dia seguinte: {future_close_pred}')

# import joblib
# model_filename = 'xgboost_model.pkl'
# joblib.dump(clf, model_filename)
# print(f'modelo salvo {model_filename}')

# from xgboost import plot_importance

# best_model = clf.best_estimator_
# plot_importance(best_model)
# plt.show()

# previsão = pd.DataFrame(index=data.index)
# previsão['Close_Real'] = data['Close']
# previsão['Previsão'] = clf.predict(X)


# previsão.loc[future_data.index[0], 'Previsão'] = future_close_pred
# previsão['Diferença'] = previsão['Previsão'] - previsão['Close_Real']

# print(previsão)

# prev_do_dia = previsão.tail(3)


# plt.figure(figsize=(10, 6))
# plt.plot(prev_do_dia.index, prev_do_dia['Close_Real'], label='Close Real')
# plt.plot(prev_do_dia.index, prev_do_dia['Previsão'], label='Previsão')
# plt.xlabel('Data')
# plt.ylabel('Valor de Fechamento')
# plt.title('Comparação dos Últimos 10 Valores de Fechamento Real e Previsão')
# plt.legend()
# plt.tight_layout()
# plt.show()



