import os
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 不用跳出視窗，專門給 Flask 在伺服器上畫圖用
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from functools import lru_cache

from model import predict_stock_price

# 1. 圖存在 static/charts/ 裡面，不要存在 static/ 根目錄
CHART_DIR = os.path.join('static', 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# 公司名字簡單對照表，不用每次都去問 Yahoo，慢又容易失敗
# 之後你可以自己加，查不到就顯示代碼就好
COMPANY_NAMES = {
    '2330': '台積電',
    '2317': '鴻海',
    '2454': '聯發科',
    '2303': '聯電',
    '2881': '富邦金',
    '2882': '國泰金',
    '2412': '中華電',
}

def _get_company_name(symbol, stock=None):
    # 先看對照表，有就直接用，不用去問 Yahoo
    if symbol in COMPANY_NAMES:
        return COMPANY_NAMES[symbol]
    # 對照表沒有，再試著去問 Yahoo，失敗就顯示代碼就好，不要讓整個查詢掛掉
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
    """用快取抓股價，同一個代碼 1 小時內再查，不用重抓"""
    # 這裡我們用簡單的 lru_cache，你之後要做更精準的 5 分鐘快取，可以用 Flask-Caching
    stock = yf.Ticker(full_symbol)
    df = stock.history(period="1y")
    return stock, df

def fetch_valid_stock(symbol):
    # 先試 .TW 上市，失敗再試 .TWO 上櫃
    # 2330 這種上市的，第一次就成功，不用白白抓第二次
    for suffix in ['.TW', '.TWO']:
        full_symbol = symbol + suffix
        try:
            stock, df = _fetch_history_cached(full_symbol)
            if df is not None and not df.empty:
                return full_symbol, stock, df
        except Exception as e:
            print(f"[WARN] 抓 {full_symbol} 失敗: {e}")
            continue
    raise ValueError(f"找不到對應的台股代碼 {symbol}（可能為下市或無資料）")

def _draw_chart(symbol, df, predicted_prices):
    """每次產生一張新的圖，檔名帶時間，不會被蓋掉"""
    plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 讓中文不會變成方塊
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index[-60:], df['Close'].iloc[-60:], label='收盤價', color='#2563eb', linewidth=1.5)
    if 'MA20' in df.columns:
        ax.plot(df.index[-60:], df['MA20'].iloc[-60:], label='MA20', color='#f59e0b', linewidth=1.2)

    # 最後一個點是今天，最後再加一個預測的點
    last_date = df.index[-1]
    next_date = last_date + __import__('pandas').Timedelta(days=1)
    ax.scatter([next_date], [predicted_prices[0]], color='#ef4444', s=60, label=f"明天預測: {predicted_prices[0]:.2f}", zorder=5)

    ax.set_title(f"{symbol} 近 60 天走勢 + 明天預測")
    ax.set_xlabel("日期")
    ax.set_ylabel("價格 (元)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    # 檔名：2330_20261119_173000.png，每次都不一樣，不會互相蓋掉
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timestamp}.png"
    filepath = os.path.join(CHART_DIR, filename)
    fig.savefig(filepath, dpi=120)
    plt.close(fig)

    # 回傳給前端的路徑，要用 /，不要用 \
    return f"static/charts/{filename}"

def analyze_stock(symbol):
    full_symbol, stock, df = fetch_valid_stock(symbol)

    # 複製一份，不要直接改到快取裡面的 df，不然下次查會越查越少
    df = df.copy()

    # 技術分析
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Returns'] = df['Close'].pct_change()

    # 只算 RSI/MACD 有值的地方，不要用 dropna 把整行砍掉
    df_valid = df.dropna(subset=['MA20', 'RSI', 'MACD'])
    if df_valid.empty:
        raise ValueError("資料不足，無法計算技術指標")

    recent_return = df_valid['Returns'].tail(7).mean()
    current_price = float(df['Close'].iloc[-1])

    # 模型預測
    predicted_prices, mape = predict_stock_price(df_valid, symbol)

    company_name = _get_company_name(symbol, stock)

    recent_text = f"""{company_name} ({symbol})
💰 目前價格：{current_price:.2f} 元
📈 近一週平均報酬率：{recent_return:.2%}
🧪 模型回測誤差（MAPE）：{mape:.2%}"""

    forecast_text = f"🔮 明天預測價格：{predicted_prices[0]:.2f} 元"

    # 真正畫圖
    chart_path = _draw_chart(symbol, df_valid, predicted_prices)

    return chart_path, recent_text, forecast_text