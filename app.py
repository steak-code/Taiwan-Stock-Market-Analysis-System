from flask import Flask, render_template, request, session
from stock_analysis import analyze_stock
import os
import re

app = Flask(__name__)
# 這個 key 是用來記你的查詢歷史，之後上線要換一個隨機的，不要用這個
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-me')

DEBUG_MODE = os.getenv('FLASK_DEBUG', 'True') == 'True'

# 熱門股，首頁會顯示
POPULAR_STOCKS = [
    {'code': '2330', 'name': '台積電', 'tag': '半導體'},
    {'code': '2317', 'name': '鴻海', 'tag': 'AI 組裝'},
    {'code': '2454', 'name': '聯發科', 'tag': 'IC 設計'},
    {'code': '2303', 'name': '聯電', 'tag': '半導體'},
    {'code': '2881', 'name': '富邦金', 'tag': '金融'},
    {'code': '2412', 'name': '中華電', 'tag': '電信'},
]

@app.route('/', methods=['GET', 'POST'])
def index():
    # 從 session 拿歷史，最多留 8 筆
    history = session.get('history', [])

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()

        if not symbol:
            return render_template('index.html', error="請輸入台股代碼", popular=POPULAR_STOCKS, history=history)

        if not re.match(r'^\d{4}$', symbol):
            return render_template('index.html', error="請輸入正確的台股代碼，例如：2330 台積電", popular=POPULAR_STOCKS, history=history)

        try:
            chart_path, backtest_path, recent_text, forecast_text, extra = analyze_stock(symbol)

            # 存到歷史，新的放最前面，重複的不要重複存
            history = [h for h in history if h['code'] != symbol]
            history.insert(0, {'code': symbol, 'name': extra.get('company_name', symbol)})
            history = history[:8]
            session['history'] = history

            # 根據 MAPE 判斷準不準
            mape = extra.get('mape', 0)
            if mape < 0.03:
                accuracy_label = "✅ 模型表現不錯"
                accuracy_color = "success"
            elif mape < 0.08:
                accuracy_label = "⚠️ 模型表現普通，參考就好"
                accuracy_color = "warning"
            else:
                accuracy_label = "❌ 模型誤差較大，請小心參考"
                accuracy_color = "danger"

            return render_template(
                'result.html',
                symbol=symbol,
                chart_path=chart_path,
                backtest_path=backtest_path,
                recent_text=recent_text,
                forecast_text=forecast_text,
                extra=extra,
                accuracy_label=accuracy_label,
                accuracy_color=accuracy_color,
                history=history
            )
        except FileNotFoundError:
            print(f"[ERROR] 模型檔案找不到，symbol={symbol}")
            return render_template('index.html', error="系統維護中，模型檔案遺失", popular=POPULAR_STOCKS, history=history)
        except ValueError as e:
            print(f"[ERROR] 查不到股票 {symbol}: {e}")
            return render_template('index.html', error=f"查不到 {symbol} 這支股票，請確認代碼是否正確", popular=POPULAR_STOCKS, history=history)
        except Exception as e:
            print(f"[ERROR] analyze_stock 發生錯誤 symbol={symbol}: {e}")
            return render_template('index.html', error="分析過程發生錯誤，請稍後再試", popular=POPULAR_STOCKS, history=history)

    return render_template('index.html', popular=POPULAR_STOCKS, history=history)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=DEBUG_MODE)