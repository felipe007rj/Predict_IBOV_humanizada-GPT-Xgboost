import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
from sqlbd import process_stock_data, update_users_news_description
import schedule
import time

def execute_trading_strategy():
    # Carregar o modelo treinado
    model_filename = 'C:\\Users\\felipe\\Downloads\\robotrade1\\xboost\\xgboost_model.pkl'
    loaded_model = joblib.load(model_filename)

    ticker = 'PETR4.SA'
    processed_data = process_stock_data(ticker)

    # Preparar o DataFrame com as features para previsão
    new_data = processed_data[['Open', 'High', 'Low', 'media_21', 'media_9', 'RSI', 'MACD', 'MACD_signal']].tail(1)

    # Pegar o último valor da coluna 'Close'
    last_close = processed_data['Close'].iloc[-1]

    # Fazer previsões usando o modelo carregado
    predicted_close = loaded_model.predict(new_data)

    # Lógica de Compra ou Venda
    if predicted_close[0] > last_close:
        signal = "Compra"
    else:
        signal = "Venda"

    # Imprimir o último valor da coluna 'Close', a previsão e a decisão
    print(f'Last close value: {last_close}')
    print(f'Predicted close value for the next day: {predicted_close[0]}')
    print(f'Signal: {signal}')

    # Armazenar os sinais em variáveis para uso posterior
    if signal == "Compra":
        compra_signal = True
        venda_signal = False
    else:
        compra_signal = False
        venda_signal = True

    # Chamar a função de atualização de descrição de notícias para usuários com base no sinal
    update_users_news_description(signal)

# Agendar a execução todos os dias úteis às 6 da manhã
schedule.every().weekday().at("06:00").do(execute_trading_strategy)

while True:
    schedule.run_pending()
    time.sleep(1)