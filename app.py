from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import tempfile
import numpy as np
from PIL import Image
import cv2
import base64
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ініціалізація Flask
app = Flask(__name__)
CORS(app)

# Конфігурація для Azure
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Lazy loading для важких модулів
depth_map_module = None
create_3dmodel_module = None

def get_depth_map():
    global depth_map_module
    if depth_map_module is None:
        try:
            from depth_map import DepthMap
            depth_map_module = DepthMap
            logger.info("DepthMap module loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import DepthMap: {e}")
            raise
    return depth_map_module

def get_image_to_3d():
    global create_3dmodel_module
    if create_3dmodel_module is None:
        try:
            from create_3dmodel import ImageTo3D
            create_3dmodel_module = ImageTo3D
            logger.info("ImageTo3D module loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import ImageTo3D: {e}")
            raise
    return create_3dmodel_module

def save_temp_file(file):
    """Збереження тимчасового файлу"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_file.name)
        logger.info(f"Temp file saved: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logger.error(f"Error saving temp file: {e}")
        raise

def process_image(temp_path, model_type='DPT_Large'):
    """Обробка зображення та генерація карти глибини"""
    try:
        image_cv = cv2.imread(temp_path)
        if image_cv is None:
            raise ValueError("Не вдалося обробити зображення")
        
        DepthMap = get_depth_map()
        depth_generator = DepthMap(model_type)
        depth_generator.load_model()
        depth_map = depth_generator.estimate_depth(image_cv)
        
        logger.info("Image processed successfully")
        return image_cv, depth_map
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def image_to_base64(image, format='.jpg'):
    """Конвертація зображення в base64"""
    try:
        _, buffer = cv2.imencode(format, image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        raise

def cleanup_file(path):
    """Видалення файлу якщо існує"""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Cleaned up file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {path}: {e}")

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({"error": "Template not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online", 
        "message": "Server is running",
        "port": os.environ.get('PORT', 'default')
    })

@app.route('/generate_3d', methods=['POST'])
def generate_3d_model():
    temp_path = None
    try:
        # Перевірка файлу
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({"error": "Файл не вибрано"}), 400

        # Отримання параметрів
        file = request.files['image']
        max_height = float(request.form.get('max_height', 20.0))
        base_thickness = float(request.form.get('base_thickness', 2.0))
        model_type = request.form.get('model_type', 'DPT_Large')
        
        logger.info(f"Processing 3D model with params: height={max_height}, thickness={base_thickness}, model={model_type}")

        # Обробка зображення
        temp_path = save_temp_file(file)
        image_cv, depth_map = process_image(temp_path, model_type)
        
        # Створення 3D моделі
        ImageTo3D = get_image_to_3d()
        model_3d = ImageTo3D()
        mask = model_3d.remove_background(image_cv, depth_map)
        mesh = model_3d.create_3d_model(depth_map, mask, max_height, base_thickness)

        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return jsonify({"error": "Не вдалося створити 3D модель"}), 500

        # Збереження OBJ файлу
        temp_obj_path = temp_path.replace('.jpg', '.obj')
        mesh.export(temp_obj_path)
        
        logger.info("3D model created successfully")

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
        logger.error(f"Error in generate_3d_model: {str(e)}")
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
        logger.error(f"Error in download_model: {str(e)}")
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
        logger.error(f"Error in process_image_only: {str(e)}")
        return jsonify({"error": f"Помилка обробки: {str(e)}"}), 500
    finally:
        if temp_path:
            cleanup_file(temp_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)