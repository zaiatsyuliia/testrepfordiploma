from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        num1 = float(data['num1'])
        num2 = float(data['num2'])
        operation = data['operation']
        
        if operation == 'add':
            result = num1 + num2
        elif operation == 'subtract':
            result = num1 - num2
        elif operation == 'multiply':
            result = num1 * num2
        elif operation == 'divide':
            if num2 == 0:
                return jsonify({'error': 'Ділення на нуль неможливе!'})
            result = num1 / num2
        else:
            return jsonify({'error': 'Невідома операція'})
        
        return jsonify({'result': result})
    
    except ValueError:
        return jsonify({'error': 'Будь ласка, введіть коректні числа'})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'OK', 'message': 'Сервер працює!'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)