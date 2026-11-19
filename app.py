from flask import Flask, render_template, request
from stock_analysis import analyze_stock
import os
import re

app = Flask(__name__)

# 只在自己電腦測試才開 debug，之後要上線要改成 False
# 可以用環境變數控制，不用每次改程式碼
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'True') == 'True'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form.get('symbol', '').strip()

        # 1. 先檢查有沒有輸入
        if not symbol:
            return render_template('index.html', error="請輸入台股代碼")

        # 2. 台股就是 4 碼數字，1 碼、5 碼直接擋掉，不用浪費 API
        if not re.match(r'^\d{4}$', symbol):
            return render_template('index.html', error="請輸入正確的台股代碼，例如：2330 台積電")

        try:
            chart_path, backtest_path, recent_text, forecast_text = analyze_stock(symbol)
            return render_template('result.html', symbol=symbol,
                        chart_path=chart_path,
                        backtest_path=backtest_path,
                        recent_text=recent_text,
                        forecast_text=forecast_text)
            
        except FileNotFoundError:
            # 模型不見了，給使用者看簡單的，給你自己看詳細的
            print(f"[ERROR] 模型檔案找不到，symbol={symbol}")
            return render_template('index.html', error="系統維護中，模型檔案遺失，請聯絡管理員")
        except ValueError as e:
            # 查不到這支股票，可能是下市、或是代碼打錯
            print(f"[ERROR] 查不到股票 {symbol}: {e}")
            return render_template('index.html', error=f"查不到 {symbol} 這支股票，請確認代碼是否正確")
        except Exception as e:
            # 其他未知錯誤，不要把真正的錯誤秀給使用者看
            print(f"[ERROR] analyze_stock 發生錯誤 symbol={symbol}: {e}")
            return render_template('index.html', error="分析過程發生錯誤，請稍後再試")

    return render_template('index.html')

if __name__ == '__main__':
    # host='0.0.0.0' 才能讓同一個 Wi-Fi 的手機也看到，不然只有自己電腦看得到
    app.run(host='127.0.0.1', port=5000, debug=DEBUG_MODE)