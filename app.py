from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import tempfile
import numpy as np
from PIL import Image
import cv2
import base64
import traceback
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')
CORS(app)

# Глобальні змінні для моделей
depth_map_instance = None
model_3d_instance = None

def initialize_models():
    """Ініціалізація моделей при старті"""
    global depth_map_instance, model_3d_instance
    try:
        from depth_map import DepthMap
        from create_3dmodel import ImageTo3D
        
        logger.info("Ініціалізація моделей...")
        depth_map_instance = DepthMap()
        model_3d_instance = ImageTo3D()
        logger.info("Моделі успішно ініціалізовані")
        return True
    except Exception as e:
        logger.error(f"Помилка ініціалізації моделей: {str(e)}")
        return False

def save_temp_file(file):
    """Збереження тимчасового файлу"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_file.name)
        logger.info(f"Тимчасовий файл збережено: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Помилка збереження файлу: {str(e)}")
        raise

def process_image(temp_path, model_type='DPT_Large'):
    """Обробка зображення та генерація карти глибини"""
    try:
        image_cv = cv2.imread(temp_path)
        if image_cv is None:
            raise ValueError("Не вдалося обробити зображення")
        
        logger.info(f"Зображення завантажено: {image_cv.shape}")
        
        # Використовуємо глобальний екземпляр або створюємо новий
        if depth_map_instance:
            depth_generator = depth_map_instance
        else:
            from depth_map import DepthMap
            depth_generator = DepthMap(model_type)
        
        if not hasattr(depth_generator, 'model') or depth_generator.model is None:
            depth_generator.load_model()
        
        depth_map = depth_generator.estimate_depth(image_cv)
        logger.info(f"Карта глибини створена: {depth_map.shape}")
        
        return image_cv, depth_map
    except Exception as e:
        logger.error(f"Помилка обробки зображення: {str(e)}")
        raise

def image_to_base64(image, format='.jpg'):
    """Конвертація зображення в base64"""
    try:
        if format == '.png':
            _, buffer = cv2.imencode('.png', image)
        else:
            _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Помилка конвертації в base64: {str(e)}")
        raise

def cleanup_file(path):
    """Видалення файлу якщо існує"""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Файл видалено: {path}")
    except Exception as e:
        logger.error(f"Помилка видалення файлу {path}: {str(e)}")

@app.route('/')
def index():
    return render_template('create_model.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "message": "Server is running",
        "models_initialized": depth_map_instance is not None and model_3d_instance is not None
    })

@app.route('/generate_3d', methods=['POST'])
def generate_3d_model():
    temp_path = None
    temp_obj_path = None
    
    try:
        # Перевірка файлу
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({"error": "Файл не вибрано"}), 400

        # Отримання параметрів
        file = request.files['image']
        max_height = float(request.form.get('max_height', 20.0))
        base_thickness = float(request.form.get('base_thickness', 2.0))
        model_type = request.form.get('model_type', 'DPT_Large')

        logger.info(f"Генерація 3D моделі: max_height={max_height}, base_thickness={base_thickness}, model={model_type}")

        # Обробка зображення
        temp_path = save_temp_file(file)
        image_cv, depth_map = process_image(temp_path, model_type)
        
        # Створення 3D моделі
        if model_3d_instance:
            model_3d = model_3d_instance
        else:
            from create_3dmodel import ImageTo3D
            model_3d = ImageTo3D()
        
        mask = model_3d.remove_background(image_cv, depth_map)
        mesh = model_3d.create_3d_model(depth_map, mask, max_height, base_thickness)

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return jsonify({"error": "Не вдалося створити 3D модель"}), 500

        # Збереження OBJ файлу
        temp_obj_path = temp_path.replace('.jpg', '.obj')
        mesh.export(temp_obj_path)
        
        logger.info(f"3D модель створена: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")

        return jsonify({
            "success": True,
            "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(),
            "vertices_count": len(mesh.vertices),
            "faces_count": len(mesh.faces),
            "image_preview": image_to_base64(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)),
            "obj_file_path": temp_obj_path,
            "message": "3D модель успішно створена"
        })

    except Exception as e:
        logger.error(f"Помилка генерації 3D моделі: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Помилка обробки: {str(e)}"}), 500
    
    finally:
        if temp_path:
            cleanup_file(temp_path)

@app.route('/download_model', methods=['POST'])
def download_model():
    try:
        data = request.get_json()
        obj_path = data.get('obj_file_path')
        
        if not obj_path or not os.path.exists(obj_path):
            return jsonify({"error": "Файл не знайдено"}), 404
        
        return send_file(obj_path, 
                        as_attachment=True, 
                        download_name='model_3d.obj',
                        mimetype='application/octet-stream')
    
    except Exception as e:
        logger.error(f"Помилка завантаження: {str(e)}")
        return jsonify({"error": f"Помилка завантаження: {str(e)}"}), 500

@app.route('/process_image', methods=['POST'])
def process_image_only():
    temp_path = None
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Не знайдено зображення"}), 400
        
        file = request.files['image']
        model_type = request.form.get('model_type', 'DPT_Large')

        temp_path = save_temp_file(file)
        image_cv, depth_map = process_image(temp_path, model_type)
        
        # Конвертація в base64
        depth_normalized = (depth_map * 255).astype(np.uint8)
        
        return jsonify({
            "success": True,
            "original_image": image_to_base64(image_cv),
            "depth_map": image_to_base64(depth_normalized, '.png'),
            "model_type": model_type
        })

    except Exception as e:
        logger.error(f"Помилка обробки зображення: {str(e)}")
        return jsonify({"error": f"Помилка обробки: {str(e)}"}), 500
    
    finally:
        if temp_path:
            cleanup_file(temp_path)

# Обробка помилок
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Ендпоінт не знайдено"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Внутрішня помилка сервера"}), 500

if __name__ == '__main__':
    # Ініціалізація моделей при старті
    models_ok = initialize_models()
    if not models_ok:
        logger.warning("Моделі не вдалося ініціалізувати, але сервер запускається")
    
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Запуск сервера на порту {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)