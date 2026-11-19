import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # ✅ 使用非互動式後端（避免使用 tkinter）
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

def predict_stock_price(df, symbol):
    # 加入技術指標
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df.dropna(inplace=True)

    feature_cols = ['Close', 'RSI', 'MACD', 'Volume']
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 分割為訓練與測試資料
    split_index = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index - 60:]

    def create_dataset(data):
        X, y = [], []
        for i in range(60, len(data)):
            X.append(data[i - 60:i])
            y.append(data[i, 0])  # 只預測 Close
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    # 建立 LSTM 模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)

    # 回測預測
    y_pred_scaled = model.predict(X_test, verbose=0)
    dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_true = np.zeros((len(y_test), len(feature_cols)))
    dummy_true[:, 0] = y_test
    y_true = scaler.inverse_transform(dummy_true)[:, 0]

    mape = mean_absolute_percentage_error(y_true, y_pred)

    # 🎯 未來 1 天預測
    last_60_days = scaled_data[-60:]
    last_input = last_60_days.reshape(1, 60, len(feature_cols))
    predicted_scaled = model.predict(last_input, verbose=0)[0][0]
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, 0] = predicted_scaled
    predicted_close = scaler.inverse_transform(dummy)[0][0]

    # =================📈 主圖：歷史價格與明日預測 =================
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    next_day = df.index[-1] + pd.Timedelta(days=1)

    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='實際收盤價', color='blue')
    plt.plot(df['MA5'], label='MA5', linestyle='--', color='orange')
    plt.plot(df['MA20'], label='MA20', linestyle='--', color='green')
    plt.plot(df['MA60'], label='MA60', linestyle='--', color='purple')
    plt.scatter([next_day], [predicted_close], label='預測明日價格', color='red', marker='o', s=100)
    plt.title(f'{symbol} 收盤價與預測（1 日）')
    plt.xlabel('時間')
    plt.ylabel('價格')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/chart.png')
    plt.close()

    # =================📉 回測圖：預測 vs 實際 =================
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='實際價格', color='blue')
    plt.plot(y_pred, label='模型預測', color='red')
    plt.title(f'{symbol} 回測預測 vs 實際收盤價')
    plt.xlabel('時間（最近 {len(y_true)} 天）')
    plt.ylabel('價格')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/backtest.png')
    plt.close()

    return [predicted_close], mape
