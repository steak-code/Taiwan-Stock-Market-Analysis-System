import os
import numpy as np
import pandas as pd
import ta
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import random
import tensorflow as tf

# ★ 固定種子，讓每次訓練結果都一樣，不然每次查都會不一樣
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ★ 關鍵：模型版本，只要改模型架構，就改這個數字，舊模型就會自動作廢
MODEL_VERSION = "v2_return"  # v1 是預測價格，v2 是預測漲跌幅

_MODEL_CACHE = {}
_SCALER_CACHE = {}

SEQ_LEN = 60  # 用過去 60 天預測明天

def _build_model(input_shape):
    """優化後的 LSTM：加入 Dropout，防止過擬合"""
    model = Sequential()
    # 第一層 LSTM，return_sequences=True，因為後面還有一層
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # 隨機丟掉 20% 的神經元，防止記住雜訊

    # 第二層 LSTM
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))

    # 輸出層：預測明天的漲跌幅，不是價格
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))  # 漲跌幅，例如 0.02 = 漲 2%

    model.compile(optimizer='adam', loss='huber')  # huber 比 mse 更不怕極端值
    return model

def _add_features(df):
    df = df.copy()
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
    df['Close'] = df['Close'].replace(0, np.nan).ffill()

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8)
    df['Daily_Return'] = df['Close'].pct_change().clip(-0.2, 0.2)
    df['Volume_Change'] = df['Volume'].pct_change().clip(-1.0, 5.0)

    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
    df['BB_Width'] = df['BB_Width'].clip(0, 1.0)

    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = df['Target'].clip(-0.1, 0.1)

    # ★ 關鍵修法：不要 dropna 把最後一天刪掉
    # 只清 inf，特徵是 NaN 的才刪，但 Target 是 NaN 的要留著，因為今天就是要拿來預測明天
    df = df.replace([np.inf, -np.inf], np.nan)

    # 只對特徵做 dropna，Target 不管
    feature_cols_for_drop = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MA5', 'MA20', 'BIAS20', 'Daily_Return', 'Volume_Change', 'BB_Width']
    feature_cols_for_drop = [c for c in feature_cols_for_drop if c in df.columns]
    df = df.dropna(subset=feature_cols_for_drop)

    return df

def _create_dataset(features, target):
    X, y = [], []
    for i in range(SEQ_LEN, len(features)):
        X.append(features[i - SEQ_LEN:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def predict_stock_price(df, symbol):
    df = _add_features(df)

    if len(df) < SEQ_LEN + 50:
        raise ValueError(f"資料不足，只有 {len(df)} 天，至少需要 {SEQ_LEN + 50} 天")

    feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'MA5', 'MA20', 'BIAS20', 'Daily_Return', 'Volume_Change', 'BB_Width']
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ★ 關鍵修法：訓練用的 df，要把最後一天 (Target 是 NaN) 拿掉
    # 預測用的 df，要留著最後一天 (今天 86.20)
    df_train = df.dropna(subset=['Target']).copy()
    df_all = df.copy()

    print(f"[DEBUG] df_all 最後一天: {df_all.index[-1].date() if hasattr(df_all.index[-1], 'date') else df_all.index[-1]} Close={df_all['Close'].iloc[-1]:.2f}")
    print(f"[DEBUG] df_train 最後一天: {df_train.index[-1].date() if hasattr(df_train.index[-1], 'date') else df_train.index[-1]} Close={df_train['Close'].iloc[-1]:.2f}")

    # ★ 切訓練 / 測試，要用 df_train，不是 df_all
    split_index = int(len(df_train) * 0.85)
    train_df = df_train.iloc[:split_index]
    test_df = df_train.iloc[split_index - SEQ_LEN:]

    # Scaler 只用訓練集來 fit
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)
    train_features = scaler.transform(train_df[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    train_target = train_df['Target'].values
    test_target = test_df['Target'].values

    X_train, y_train = _create_dataset(train_features, train_target)
    X_test, y_test = _create_dataset(test_features, test_target)

    model_path = os.path.join(MODEL_DIR, f"{symbol}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    version_path = os.path.join(MODEL_DIR, f"{symbol}_version.txt")

    # ★ 檢查版本，舊模型自動作廢
    need_retrain = True
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(version_path):
        try:
            with open(version_path, 'r') as f:
                saved_version = f.read().strip()
            if saved_version == MODEL_VERSION:
                need_retrain = False
            else:
                print(f"[WARN] {symbol} 模型版本不對 {saved_version} != {MODEL_VERSION}，自動重訓")
        except:
            need_retrain = True

    if symbol in _MODEL_CACHE and not need_retrain:
        model = _MODEL_CACHE[symbol]
        print(f"[INFO] 從快取使用 {symbol} 模型")
    elif not need_retrain:
        print(f"[INFO] 從硬碟載入 {symbol} 模型")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        _MODEL_CACHE[symbol] = model
        _SCALER_CACHE[symbol] = scaler
    else:
        print(f"[INFO] 訓練新的 {symbol} 模型...")
        model = _build_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        with open(version_path, 'w') as f:
            f.write(MODEL_VERSION)
        _MODEL_CACHE[symbol] = model
        _SCALER_CACHE[symbol] = scaler
        print(f"[INFO] {symbol} 模型已儲存")

    # 預測漲跌幅 (測試集)
    y_pred_return = model.predict(X_test, verbose=0).flatten()
    y_pred_return = np.clip(y_pred_return, -0.1, 0.1)

    # 還原成價格
    today_close = test_df['Close'].values[SEQ_LEN - 1:-1]
    y_pred_price = today_close * (1 + y_pred_return)
    y_true_price_aligned = test_df['Close'].values[SEQ_LEN:]

    min_len = min(len(y_pred_price), len(y_true_price_aligned))
    y_pred_price = y_pred_price[:min_len]
    y_true_price_aligned = y_true_price_aligned[:min_len]
    y_pred_return = y_pred_return[:min_len]
    y_test = y_test[:min_len]

    # MAPE
    mape = mean_absolute_percentage_error(y_true_price_aligned, y_pred_price)
    naive_pred_price = today_close[:min_len]
    naive_mape = mean_absolute_percentage_error(y_true_price_aligned, naive_pred_price)

    # 方向準確率
    true_direction = (y_true_price_aligned > today_close[:min_len]).astype(int)
    pred_direction = (y_pred_price > today_close[:min_len]).astype(int)
    direction_acc = (true_direction == pred_direction).mean()

    print(f"[INFO] {symbol} 模型 MAPE: {mape:.2%}, 天真 MAPE: {naive_mape:.2%}, 方向準確率: {direction_acc:.2%}")

    # ★ 明天預測：要用 df_all 的最後 60 天，包含今天 86.20
    # 不能用 df_train 的最後 60 天，那會是昨天 95.70
    last_60 = scaler.transform(df_all[feature_cols].values[-SEQ_LEN:])
    last_input = last_60.reshape(1, SEQ_LEN, len(feature_cols))
    predicted_return = float(model.predict(last_input, verbose=0)[0][0])

    if abs(predicted_return) > 1.0:
        print(f"[WARN] {symbol} 模型輸出異常 {predicted_return}，可能是舊模型，已強制歸零")
        predicted_return = 0.0

    predicted_return = float(np.clip(predicted_return, -0.1, 0.1))

    # ★ 這裡的 last_price 現在是今天的 86.20，不是昨天的 95.70
    last_price = float(df_all['Close'].iloc[-1])
    predicted_close = last_price * (1 + predicted_return)
    predicted_close = float(np.clip(predicted_close, last_price * 0.9, last_price * 1.1))

    print(f"[INFO] {symbol} 目前 {last_price:.2f}，預測漲跌幅 {predicted_return:.2%}，預測價 {predicted_close:.2f}，漲停 {last_price*1.1:.2f}，跌停 {last_price*0.9:.2f}")

    return [predicted_close], mape, y_true_price_aligned, y_pred_price, {
        'direction_acc': float(direction_acc),
        'naive_mape': float(naive_mape),
        'predicted_return': predicted_return,
        'is_better_than_naive': bool(mape < naive_mape),
        'last_price': last_price,
    }