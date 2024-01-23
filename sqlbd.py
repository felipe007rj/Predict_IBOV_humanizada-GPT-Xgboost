import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
import random
import openai

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

##################  Resposta humanizada pelo CHATGPT  ########################

# def update_users_news_description(signal):
#     # Conecte-se ao banco de dados
#     conn = sqlite3.connect('teste.db')
#     cursor = conn.cursor()

#     # Obter os registros que atendem ao critério
#     cursor.execute(f'SELECT id, name FROM users WHERE poupança > "9000";')
#     rows = cursor.fetchall()

#     # Atualizar a coluna news_description com base no sinal e no nome do usuário
#     for row in rows:
#         user_id = row[0]
#         user_name = row[1]

#         # Variações para {user_name} e {signal}
#         user_name_variations = [user_name, f"{user_name} amigo", f"{user_name} investidor"]
#         signal_variations = [signal, "compra", "venda"]

#         # Escolher aleatoriamente variações
#         selected_user_name = random.choice(user_name_variations)
#         selected_signal = random.choice(signal_variations)

#         # Criar uma mensagem com variações
#         prompt = f"Olá, {selected_user_name}! De acordo com a inteligência artificial, hoje é um bom dia para {selected_signal} de ações da Petrobras."

#         # Usar a API do ChatGPT para obter uma resposta
#         response = openai.Completion.create(
#             engine="text-davinci-002",
#             prompt=prompt,
#             temperature=0.7,
#             max_tokens=150
#         )

#         # Obter a resposta da API e atualizar a descrição de notícias no banco de dados
#         generated_response = response.choices[0].text.strip()

#         # Adicionar a resposta gerada à descrição final
#         news_description = f"{generated_response}"

#         cursor.execute(f'UPDATE users SET news_description = ? WHERE id = ?', (news_description, user_id))

#     conn.commit()
#     cursor.close()
#     conn.close()


if __name__ == "__main__":
    update_users_news_description()

if __name__ == "__main__":
    # Executar apenas a função process_stock_data
    ticker = 'PETR4.SA'
    processed_data = process_stock_data(ticker)
    print(processed_data)

