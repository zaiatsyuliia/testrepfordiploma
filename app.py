from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import tempfile
import numpy as np
from PIL import Image
import cv2
import base64
from depth_map import DepthMap
from create_3dmodel import ImageTo3D

app = Flask(__name__, static_folder='static')
CORS(app)
model_3d = ImageTo3D()

def save_temp_file(file):
    """Збереження тимчасового файлу"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    file.save(temp_file.name)
    return temp_file.name

def process_image(temp_path, model_type='DPT_Large'):
    """Обробка зображення та генерація карти глибини"""
    image_cv = cv2.imread(temp_path)
    if image_cv is None:
        raise ValueError("Не вдалося обробити зображення")
    
    depth_generator = DepthMap(model_type)
    depth_generator.load_model()
    depth_map = depth_generator.estimate_depth(image_cv)
    return image_cv, depth_map

def image_to_base64(image, format='.jpg'):
    """Конвертація зображення в base64"""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode('utf-8')

def cleanup_file(path):
    """Видалення файлу якщо існує"""
    if os.path.exists(path):
        os.unlink(path)

@app.route('/')
def index():
    return render_template('create_model.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Server is running"})

@app.route('/generate_3d', methods=['POST'])
def generate_3d_model():
    try:
        # Перевірка файлу
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({"error": "Файл не вибрано"}), 400

        # Отримання параметрів
        file = request.files['image']
        max_height = float(request.form.get('max_height', 20.0))
        base_thickness = float(request.form.get('base_thickness', 2.0))
        model_type = request.form.get('model_type', 'DPT_Large')

        # Обробка зображення
        temp_path = save_temp_file(file)
        try:
            image_cv, depth_map = process_image(temp_path, model_type)
            
            # Створення 3D моделі
            mask = model_3d.remove_background(image_cv, depth_map)
            mesh = model_3d.create_3d_model(depth_map, mask, max_height, base_thickness)

            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                return jsonify({"error": "Не вдалося створити 3D модель"}), 500

            # Збереження OBJ файлу
            temp_obj_path = temp_path.replace('.jpg', '.obj')
            mesh.export(temp_obj_path)

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

        finally:
            cleanup_file(temp_path)

    except Exception as e:
        print(f"Помилка: {str(e)}")
        return jsonify({"error": f"Помилка обробки: {str(e)}"}), 500

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
        return jsonify({"error": f"Помилка завантаження: {str(e)}"}), 500

@app.route('/process_image', methods=['POST'])
def process_image_only():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Не знайдено зображення"}), 400
        
        file = request.files['image']
        model_type = request.form.get('model_type', 'DPT_Large')

        temp_path = save_temp_file(file)
        try:
            image_cv, depth_map = process_image(temp_path, model_type)
            
            # Конвертація в base64
            depth_normalized = (depth_map * 255).astype(np.uint8)
            
            return jsonify({
                "success": True,
                "original_image": image_to_base64(image_cv),
                "depth_map": image_to_base64(depth_normalized, '.png'),
                "model_type": model_type
            })

        finally:
            cleanup_file(temp_path)

    except Exception as e:
        return jsonify({"error": f"Помилка обробки: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)