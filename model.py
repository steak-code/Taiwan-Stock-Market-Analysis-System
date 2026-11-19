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
    """特徵工程：加入防 inf 清理"""
    df = df.copy()

    # 先把 Volume 的 0 換掉，不然 pct_change 會變 inf
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill().fillna(1)
    df['Close'] = df['Close'].replace(0, np.nan).ffill()

    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()

    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    # ★ 修法：分母加一個很小的值，不會除以 0
    df['BIAS20'] = (df['Close'] - df['MA20']) / (df['MA20'] + 1e-8)

    # ★ 修法：pct_change 後，限制在 -20% 到 +20%，台股漲跌停是 10%，超過就是髒資料
    df['Daily_Return'] = df['Close'].pct_change().clip(-0.2, 0.2)
    df['Volume_Change'] = df['Volume'].pct_change().clip(-1.0, 5.0)  # 量最多放大 5 倍

    bb = ta.volatility.BollingerBands(df['Close'])
    # ★ 修法：分母加 1e-8，不會除以 0
    df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['Close'] + 1e-8)
    df['BB_Width'] = df['BB_Width'].clip(0, 1.0)

    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = df['Target'].clip(-0.1, 0.1)  # 台股漲跌停 10%

    # ★ 最關鍵：把所有 inf 換成 NaN，再全部刪掉
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 再檢查一次，確保沒有 inf
    # 如果還有，就把那一行印出來，方便你 debug
    if np.isinf(df.select_dtypes(include=[np.number]).values).any():
        print("[ERROR] 還有 inf，清不乾淨")
        df = df.replace([np.inf, -np.inf], 0)

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

    # ★ 特徵清單，8 個特徵
    feature_cols = [
        'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal',
        'MA5', 'MA20', 'BIAS20',
        'Daily_Return', 'Volume_Change', 'BB_Width'
    ]
    # 確保都有，不會因為某個指標是 NaN 就掛掉
    feature_cols = [c for c in feature_cols if c in df.columns]

    # ★ 關鍵修正：切訓練 / 測試，要先切，再算 Scaler
    # 不然會用未來的資料來算 Scaler，就是偷看未來
    split_index = int(len(df) * 0.85)  # 85% 訓練，15% 測試，比之前留更多測試
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index - SEQ_LEN:]  # 測試要多留 60 天，才能做出第一個序列

    # Scaler 只用訓練集來 fit，測試集只能 transform
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols].values)

    # 轉換
    train_features = scaler.transform(train_df[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    # 目標：明天漲跌幅
    train_target = train_df['Target'].values
    test_target = test_df['Target'].values

    X_train, y_train = _create_dataset(train_features, train_target)
    X_test, y_test = _create_dataset(test_features, test_target)

    # 實際價格，用來算 MAPE 和方向
    # test_df 的 Close，要對齊 X_test
    test_close = test_df['Close'].values[SEQ_LEN:]
    # y_test 是漲跌幅，實際明天價格 = 今天價格 * (1 + 漲跌幅)
    # 這裡我們用 test_close 的前一天，來還原
    # 簡單版：y_true_price = test_close[1:]，因為 test_close 已經是明天了
    # 為了不搞混，我們直接用 test_df 的 Close 來算
    y_true_price = test_df['Close'].values[SEQ_LEN:]

    model_path = os.path.join(MODEL_DIR, f"{symbol}.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    version_path = os.path.join(MODEL_DIR, f"{symbol}_version.txt")

    # ★ 檢查版本，如果版本不對，或是沒有版本檔，就是舊模型，直接刪掉重訓
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
        # ★ 存版本檔
        with open(version_path, 'w') as f:
            f.write(MODEL_VERSION)
        _MODEL_CACHE[symbol] = model
        _SCALER_CACHE[symbol] = scaler
        print(f"[INFO] {symbol} 模型已儲存")

    # 預測漲跌幅
    y_pred_return = model.predict(X_test, verbose=0).flatten()
    y_pred_return = np.clip(y_pred_return, -0.1, 0.1)  # 台股漲跌停 10%，超過就砍掉，不讓它亂猜 20%

    # 還原成價格：明天價格 = 今天價格 * (1 + 漲跌幅)
    # 今天價格是 test_close 的前一天
    # 為了簡單，我們用 y_true_price 的前一天當今天
    # test_close 是 [今天, 明天, ...]，所以要往前推
    # 這裡用 test_df 的 Close 來對齊
    today_close = test_df['Close'].values[SEQ_LEN - 1:-1]  # 今天
    y_pred_price = today_close * (1 + y_pred_return)
    y_true_price_aligned = test_df['Close'].values[SEQ_LEN:]  # 明天實際

    # 確保長度一樣
    min_len = min(len(y_pred_price), len(y_true_price_aligned))
    y_pred_price = y_pred_price[:min_len]
    y_true_price_aligned = y_true_price_aligned[:min_len]
    y_pred_return = y_pred_return[:min_len]
    y_test = y_test[:min_len]

    # MAPE
    mape = mean_absolute_percentage_error(y_true_price_aligned, y_pred_price)

    # ★ 天真預測：猜明天 = 今天，拿來比較
    naive_pred_price = today_close[:min_len]
    naive_mape = mean_absolute_percentage_error(y_true_price_aligned, naive_pred_price)

    # ★ 方向準確率：猜對漲或跌的機率，這個比 MAPE 更重要
    # 實際漲跌：明天 > 今天 = 漲
    true_direction = (y_true_price_aligned > today_close[:min_len]).astype(int)
    pred_direction = (y_pred_price > today_close[:min_len]).astype(int)
    direction_acc = (true_direction == pred_direction).mean()

    print(f"[INFO] {symbol} 模型 MAPE: {mape:.2%}, 天真 MAPE: {naive_mape:.2%}, 方向準確率: {direction_acc:.2%}")

        # 明天預測：用最近 60 天
    last_60 = scaler.transform(df[feature_cols].values[-SEQ_LEN:])
    last_input = last_60.reshape(1, SEQ_LEN, len(feature_cols))
    predicted_return = float(model.predict(last_input, verbose=0)[0][0])

    # ★ 第一道保險：漲跌幅限制在 -10% 到 +10%
    # 如果模型輸出是 0.147，或是 98.88 這種離譜的數字，直接砍掉
    # 如果它輸出的是價格 (例如 98)，那它一定 > 0.1，就會被砍成 0.1
    # 這樣就不會再出現 14.7% 這種事
    if abs(predicted_return) > 1.0:
        print(f"[WARN] {symbol} 模型輸出異常 {predicted_return}，可能是舊模型，已強制歸零")
        predicted_return = 0.0

    predicted_return = float(np.clip(predicted_return, -0.1, 0.1))
    last_price = float(df['Close'].iloc[-1])

    # ★ 第二道保險：價格也砍一次，保證在漲跌停內
    predicted_close = last_price * (1 + predicted_return)
    predicted_close = float(np.clip(predicted_close, last_price * 0.9, last_price * 1.1))

    print(f"[INFO] {symbol} 目前 {last_price:.2f}，預測漲跌幅 {predicted_return:.2%}，預測價 {predicted_close:.2f}，漲停 {last_price*1.1:.2f}，跌停 {last_price*0.9:.2f}")

    # 回傳：預測價格、MAPE、實際、預測、還有方向準確率和天真 MAPE
    return [predicted_close], mape, y_true_price_aligned, y_pred_price, {
        'direction_acc': float(direction_acc),
        'naive_mape': float(naive_mape),
        'predicted_return': predicted_return,
        'is_better_than_naive': bool(mape < naive_mape),
    }