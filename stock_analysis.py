import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import pandas as pd
from functools import lru_cache
from model import predict_stock_price
import logging
logging.getLogger('yfinance').setLevel(logging.ERROR)

CHART_DIR = os.path.join('static', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

COMPANY_NAMES = {
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科',
    '2303': '聯電', '2881': '富邦金', '2882': '國泰金', '2412': '中華電',
}

def _get_company_name(symbol, stock=None):
    if symbol in COMPANY_NAMES:
        return COMPANY_NAMES[symbol]
    try:
        if stock is not None:
            name = stock.info.get('longName') or stock.info.get('shortName')
            if name:
                return name
    except Exception as e:
        print(f"[WARN] 抓公司名失敗 {symbol}: {e}")
    return f"台股 {symbol}"

@lru_cache(maxsize=32)
def _fetch_history_cached(full_symbol):
    stock = yf.Ticker(full_symbol)
    df = stock.history(period="1y")
    return stock, df

def fetch_valid_stock(symbol):
    last_error = None
    for suffix in ['.TW', '.TWO']:
        full_symbol = symbol + suffix
        try:
            stock, df = _fetch_history_cached(full_symbol)

            # ★ 關鍵：資料太少就不算成功，繼續試下一個
            if df is None or df.empty or len(df) < 100:
                continue

            # ★ 最後一天收盤價不能是 0 或 NaN
            last_close = df['Close'].iloc[-1]
            if last_close is None or pd.isna(last_close) or float(last_close) <= 0:
                continue

            # 成功了，才回傳，不要印任何 WARNING
            # print(f"[INFO] 找到 {full_symbol}，共 {len(df)} 筆")
            return full_symbol, stock, df

        except Exception as e:
            # ★ 先不要印，把錯誤存起來就好
            # 如果 .TW 失敗，我們還要試 .TWO，不用嚇使用者
            last_error = e
            continue

    # ★ 兩個都失敗，才真的印一次
    print(f"[WARN] 抓 {symbol} 失敗，.TW 和 .TWO 都查不到：{last_error}")
    raise ValueError(f"找不到對應的台股代碼 {symbol}，請確認代碼是否正確")

def _draw_charts(symbol, df, predicted_close, y_true, y_pred):
    plt.rcParams['font.family'] = 'Microsoft JhengHei'
    plt.rcParams['axes.unicode_minus'] = False
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 主圖
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-60:], df['Close'].iloc[-60:], label='收盤價', color='#2563eb', linewidth=1.6)
    ax.plot(df.index[-60:], df['MA20'].iloc[-60:], label='MA20', color='#f59e0b', linewidth=1.2)
    next_day = df.index[-1] + pd.Timedelta(days=1)
    ax.scatter([next_day], [predicted_close], color='#ef4444', s=90, label=f"明天預測: {predicted_close:.2f}", zorder=5)
    ax.set_title(f"{symbol} 近 60 天走勢", fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); fig.autofmt_xdate(); fig.tight_layout()
    chart_name = f"{symbol}_{timestamp}.png"
    fig.savefig(os.path.join(CHART_DIR, chart_name), dpi=130); plt.close(fig)

    # 回測圖
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(y_true, label='實際價格', color='#2563eb')
    ax2.plot(y_pred, label='模型預測', color='#ef4444')
    ax2.set_title(f"{symbol} 回測預測 vs 實際 (最近 {len(y_true)} 天)", fontsize=12, fontweight='bold')
    ax2.legend(); ax2.grid(alpha=0.3); fig2.tight_layout()
    backtest_name = f"{symbol}_{timestamp}_backtest.png"
    fig2.savefig(os.path.join(CHART_DIR, backtest_name), dpi=130); plt.close(fig2)

    return f"static/charts/{chart_name}", f"static/charts/{backtest_name}"

def analyze_stock(symbol):
    full_symbol, stock, df = fetch_valid_stock(symbol)
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Returns'] = df['Close'].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan)
    df_valid = df.dropna(subset=['MA20', 'RSI', 'MACD'])
    if df_valid.empty:
        raise ValueError("資料不足，無法計算技術指標")

    recent_return = float(df_valid['Returns'].tail(7).mean())
    current_price = float(df['Close'].iloc[-1])

    predicted_prices, mape, y_true, y_pred, model_info = predict_stock_price(df_valid, symbol)

    # ★ 第三道保險：台股漲跌停 10%，絕對不能超過
    raw_predicted = float(predicted_prices[0])
    limit_up = current_price * 1.1
    limit_down = current_price * 0.9

    if raw_predicted > limit_up or raw_predicted < limit_down:
        print(f"[WARN] {symbol} 預測價 {raw_predicted:.2f} 超出漲跌停 [{limit_down:.2f}, {limit_up:.2f}]，已強制修正為 {max(min(raw_predicted, limit_up), limit_down):.2f}")

    predicted_price = float(np.clip(raw_predicted, limit_down, limit_up))

    company_name = _get_company_name(symbol, stock)

    recent_text = f"""{company_name} ({symbol})
💰 目前價格：{current_price:.2f} 元
📈 近一週平均報酬率：{recent_return:.2%}
🧪 模型回測誤差（MAPE）：{float(mape):.2%}"""

    forecast_text = f"🔮 明天預測價格：{predicted_price:.2f} 元"

    extra = {
        'company_name': company_name,
        'current_price': current_price,
        'recent_return': recent_return,
        'mape': float(mape),
        'rsi': float(df_valid['RSI'].iloc[-1]),
        'macd': float(df_valid['MACD'].iloc[-1]),
        'ma20': float(df_valid['MA20'].iloc[-1]),
        'predicted_price': predicted_price,  # ★ 用修正過的，不是 raw 的
        'limit_up': limit_up,   # ★ 新增，給前端顯示漲停價
        'limit_down': limit_down,
        'direction_acc': model_info.get('direction_acc', 0),
        'naive_mape': model_info.get('naive_mape', 0),
        'predicted_return': model_info.get('predicted_return', 0),
        'is_better_than_naive': model_info.get('is_better_than_naive', False),
    }

    chart_path, backtest_path = _draw_charts(symbol, df_valid, predicted_price, y_true, y_pred)

    return chart_path, backtest_path, recent_text, forecast_text, extra