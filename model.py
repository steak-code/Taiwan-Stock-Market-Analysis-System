import os
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

# 模型存在這裡，不要每次都重訓
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 記憶體裡快取模型，同一支股票第二次查，不用重讀硬碟
_MODEL_CACHE = {}

def _build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def _prepare_data(df):
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df = df.dropna()

    feature_cols = ['Close', 'RSI', 'MACD', 'Volume']
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return df, scaled_data, scaler, feature_cols

def _create_dataset(data):
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i - 60:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def predict_stock_price(df, symbol):
    df, scaled_data, scaler, feature_cols = _prepare_data(df)

    split_index = int(len(scaled_data) * 0.9)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index - 60:]

    X_train, y_train = _create_dataset(train_data)
    X_test, y_test = _create_dataset(test_data)

    # 1. 先看記憶體有沒有，有就直接用
    # 2. 再看硬碟有沒有 .h5，有就直接載入，不用重訓
    # 3. 都沒有，才訓練一次，然後存起來
    model_path = os.path.join(MODEL_DIR, f"{symbol}.h5")
    if symbol in _MODEL_CACHE:
        model = _MODEL_CACHE[symbol]
        print(f"[INFO] 從快取使用 {symbol} 模型")
    elif os.path.exists(model_path):
        print(f"[INFO] 從硬碟載入 {symbol} 模型: {model_path}")
        model = load_model(model_path)
        _MODEL_CACHE[symbol] = model
    else:
        print(f"[INFO] 訓練新的 {symbol} 模型...")
        model = _build_model((X_train.shape[1], X_train.shape[2]))
        # 第一次訓練 epochs 可以少一點，10 就夠了，30 太慢
        # 之後要更準，再調回來
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        model.save(model_path)
        _MODEL_CACHE[symbol] = model
        print(f"[INFO] {symbol} 模型已儲存")

    # 回測
    y_pred_scaled = model.predict(X_test, verbose=0)
    dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_true = np.zeros((len(y_test), len(feature_cols)))
    dummy_true[:, 0] = y_test
    y_true = scaler.inverse_transform(dummy_true)[:, 0]

    mape = mean_absolute_percentage_error(y_true, y_pred)

    # 明天預測
    last_60_days = scaled_data[-60:]
    last_input = last_60_days.reshape(1, 60, len(feature_cols))
    predicted_scaled = model.predict(last_input, verbose=0)[0][0]
    dummy = np.zeros((1, len(feature_cols)))
    dummy[0, 0] = predicted_scaled
    predicted_close = scaler.inverse_transform(dummy)[0][0]

    # 只回傳數字，不要在這裡畫圖
    # 畫圖交給 stock_analysis.py 統一處理
    return [predicted_close], mape, y_true, y_pred