import yfinance as yf
import ta
from model import predict_stock_price

def fetch_valid_stock(symbol):
    for suffix in ['.TW', '.TWO']:
        full_symbol = symbol + suffix
        stock = yf.Ticker(full_symbol)
        df = stock.history(period="1y")
        if not df.empty:
            return full_symbol, stock, df
    raise ValueError("❌ 找不到對應的台股代碼（可能為下市或無資料）")

def analyze_stock(symbol):
    full_symbol, stock, df = fetch_valid_stock(symbol)

    # 技術分析
    df['MA20'] = df['Close'].rolling(20).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()

    df.dropna(inplace=True)
    df['Returns'] = df['Close'].pct_change()
    recent_return = df['Returns'][-7:].mean()
    current_price = df['Close'].iloc[-1]

    # 模型預測與準確度回傳
    predicted_prices, mape = predict_stock_price(df, symbol)

    company_name = stock.info.get('longName', '未知公司')

    recent_text = f"""{company_name}
   💰 目前價格：{current_price:.2f} 元
   📈 近一週平均報酬率：{recent_return:.2%}
   🧪 模型回測誤差（MAPE）：{mape:.2%}"""

    forecast_text = f"🔮 明天預測價格：{predicted_prices[0]:.2f} 元"

    return 'static/chart.png', recent_text, forecast_text
