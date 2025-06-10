import os
import argparse
import numpy as np
import torch
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import cv2
from depth_map import DepthMap
from sklearn.cluster import KMeans

class ImageTo3D:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Ініціалізація класу з моделями глибокого навчання."""
        self.device = device
        print(f"Використовуємо пристрій: {self.device}")
        
        # Завантаження моделі для оцінки глибини на основі DPT
        print("Завантаження моделі DPT для оцінки глибини...")
        try:
            self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
            self.depth_model.to(self.device)
            self.depth_model.eval()

            # Завантаження обробника вхідних даних для моделі
            self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
            print("Моделі глибокого навчання успішно завантажені")
        except:
            print("Не вдалося завантажити DPT модель, використовуємо альтернативний метод")
            self.depth_model = None
            self.feature_extractor = None
    
    def remove_background(self, image, depth_map):
        """Комплексне видалення фону"""
        print("Видалення фону...")
        
        # GrabCut
        h, w = image.shape[:2]
        margin_x, margin_y = int(w * 0.1), int(h * 0.1)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        mask = np.zeros((h, w), np.uint8)
        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask_grabcut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        except:
            mask_grabcut = np.ones((h, w), dtype=np.uint8)
        
        # Колірна сегментація
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            edge_size = min(20, w//10, h//10)
            edges = [image_rgb[:edge_size, :], image_rgb[-edge_size:, :],
                    image_rgb[:, :edge_size], image_rgb[:, -edge_size:]]
            edge_pixels = np.vstack([e.reshape(-1, 3) for e in edges])
            
            try:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                bg_color = kmeans.fit(edge_pixels).cluster_centers_[
                    np.argmax(np.bincount(kmeans.labels_))]
                distances = np.linalg.norm(image_rgb.reshape(-1, 3) - bg_color, axis=1)
                mask_color = (distances > np.percentile(distances, 30)).reshape(h, w).astype(np.uint8)
            except:
                mask_color = np.ones((h, w), dtype=np.uint8)
        else:
            mask_color = np.ones((h, w), dtype=np.uint8)
        
        # Глибинна сегментація
        mask_depth = (depth_map > np.percentile(depth_map, 60)).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_depth = cv2.morphologyEx(cv2.morphologyEx(mask_depth, cv2.MORPH_CLOSE, kernel), 
                                     cv2.MORPH_OPEN, kernel)
        
        # Комбінування і очистка
        combined = (mask_grabcut * 0.4 + mask_color * 0.3 + mask_depth * 0.3 > 0.5).astype(np.uint8)
        
        # Морфологічна очистка
        kernel_small, kernel_large = np.ones((3, 3), np.uint8), np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small), 
                                   cv2.MORPH_CLOSE, kernel_large)
        
        # Найбільша компонента
        _, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
        if stats.shape[0] > 1:
            largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            combined = (labels == largest).astype(np.uint8)
        
        return combined

    def create_3d_model(self, depth_map, mask, max_height=20.0, base_thickness=2.0):
        """Створення водонепроникної 3D моделі"""
        print("Створення 3D-моделі...")
        
        masked_depth = depth_map * mask
        rows, cols = np.where(mask > 0)
        
        if len(rows) == 0:
            mask = np.ones(depth_map.shape, dtype=np.uint8)
            rows, cols = np.where(mask > 0)
        
        # Обрізка
        margin = 2
        min_row = max(0, rows.min() - margin)
        max_row = min(depth_map.shape[0], rows.max() + margin)
        min_col = max(0, cols.min() - margin)
        max_col = min(depth_map.shape[1], cols.max() + margin)
        
        cropped_depth = masked_depth[min_row:max_row, min_col:max_col]
        cropped_mask = mask[min_row:max_row, min_col:max_col]
        
        return self.create_mesh(cropped_depth, cropped_mask, max_height, base_thickness)

    def create_mesh(self, depth_map, mask, max_height, base_thickness):
        """Створення mesh"""
        rows, cols = depth_map.shape
        
        # Масштабування глибини
        non_zero = depth_map[mask > 0]
        if len(non_zero) > 0:
            depth_min, depth_max = non_zero.min(), non_zero.max()
            scaled_depth = ((depth_map - depth_min) / (depth_max - depth_min) * max_height
                            if depth_max > depth_min else depth_map * max_height)
        else:
            scaled_depth = depth_map * max_height
        
        vertices = []
        faces = []
        vertex_map_front = {}
        vertex_map_back = {}
        vertex_idx = 0

        # Створення вершин (передня і задня частини)
        for i in range(rows):
            for j in range(cols):
                if mask[i, j] > 0:
                    x = j
                    y = rows - 1 - i
                    z = scaled_depth[i, j] + base_thickness
                    # Передня сторона
                    vertices.append([x, y, z])
                    vertex_map_front[(i, j)] = vertex_idx
                    vertex_idx += 1
                    # Задня сторона (дзеркальна по Z)
                    vertices.append([x, y, -z])
                    vertex_map_back[(i, j)] = vertex_idx
                    vertex_idx += 1

        if not vertices:
            return trimesh.Trimesh()

        vertices = np.array(vertices)

        # Поверхні: перед і зад
        for flip, vertex_map in [(False, vertex_map_front), (True, vertex_map_back)]:
            for i in range(rows - 1):
                for j in range(cols - 1):
                    corners = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]
                    if all(mask[r, c] > 0 for r, c in corners):
                        v = [vertex_map.get(pos) for pos in corners]
                        if all(x is not None for x in v):
                            if flip:
                                # Дзеркально (задня частина)
                                faces.extend([[v[0], v[2], v[1]], [v[1], v[2], v[3]]])
                            else:
                                faces.extend([[v[0], v[1], v[2]], [v[1], v[3], v[2]]])

        # Бічні стінки (з'єднання передньої і задньої частин)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            for i in range(len(contour)):
                p1 = tuple(contour[i][0])
                p2 = tuple(contour[(i + 1) % len(contour)][0])
                pos1 = (p1[1], p1[0])
                pos2 = (p2[1], p2[0])
                if all(pos in vertex_map_front and pos in vertex_map_back for pos in [pos1, pos2]):
                    v1f = vertex_map_front[pos1]
                    v2f = vertex_map_front[pos2]
                    v1b = vertex_map_back[pos1]
                    v2b = vertex_map_back[pos2]
                    faces.extend([[v1f, v2f, v1b], [v2f, v2b, v1b]])

        if not faces:
            return trimesh.Trimesh(vertices=vertices)

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.vertices -= mesh.centroid

        return mesh

    
    def process_and_save(self, image_path, output_path):
        """Основна обробка"""
        # Завантаження
        image = Image.open(image_path).convert('RGB')
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            raise ValueError(f"Не вдалося завантажити: {image_path}")
        
        # Генерація глибини
        depth_map_generator = DepthMap("DPT_Large")
        depth_map_generator.load_model()
        depth_map = depth_map_generator.estimate_depth(image_cv)
        
        # Видалення фону
        mask = self.remove_background(image_cv, depth_map)
        
        # 3D модель
        mesh = self.create_3d_model(depth_map, mask)
        
        # Збереження
        if len(mesh.vertices) > 0 and len(mesh.faces) > 0:
            mesh.export(output_path)
            print(f"3D-модель збережена: {output_path}")
            print(f"Вершини: {len(mesh.vertices)}, Грані: {len(mesh.faces)}")
        else:
            print("Не вдалося створити модель")
        
        # Візуалізація
        self.visualize(image, depth_map, mask, mesh)
        self.visualize_results(image, depth_map, mask, mesh)
        return mesh
    
    def visualize(self, image, depth_map, mask, mesh):
        """Візуалізація"""
        plt.figure(figsize=(16, 4))
        
        plt.subplot(1, 4, 1)
        plt.title("Оригінальне зображення")
        plt.imshow(image)
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.title("Карта глибини")
        plt.imshow(depth_map, cmap='plasma')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.title("Маска об'єкта")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.title("Об'єкт без фону")
        segmented = np.array(image) * mask[:,:,np.newaxis]
        plt.imshow(segmented)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualize_segmentation_result.png', dpi=150, bbox_inches='tight')
        print("Результат сегментації збережено: visualize_segmentation_result.png")

    def visualize_results(self, original_image, depth_map, mask, mesh):
        """Візуалізація результатів."""
        plt.figure(figsize=(20, 6))
        
        # Вхідне зображення
        plt.subplot(1, 4, 1)
        plt.title("Вхідне зображення")
        plt.imshow(original_image)
        plt.axis('off')
        
        # Маска об'єкта
        plt.subplot(1, 4, 2)
        plt.title("Виділений об'єкт")
        segmented = np.array(original_image) * mask[:,:,np.newaxis]
        plt.imshow(segmented)
        plt.axis('off')
        
        # Карта глибини з маскою
        plt.subplot(1, 4, 3)
        plt.title("Карта глибини об'єкта")
        masked_depth = depth_map * mask
        depth_viz = plt.imshow(masked_depth, cmap='plasma')
        plt.colorbar(depth_viz, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 3D проекція
        plt.subplot(1, 4, 4)
        plt.title("3D модель")
        if len(mesh.vertices) > 0:
            verts = mesh.vertices
            xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
            plt.scatter(xs, ys, c=zs, cmap='plasma', s=0.5, alpha=0.6)
            plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        
        plt.tight_layout()
        plt.savefig('visualize_3d_result.png', dpi=150, bbox_inches='tight')
        print("Результат збережено: visualize_3d_result.png")
        
        # Створення 3D рендеру
        try:
            if len(mesh.vertices) > 0:
                scene = trimesh.Scene(mesh)
                
                # Налаштовуємо камеру
                scene.camera_transform = trimesh.transformations.rotation_matrix(
                    np.radians(-30), [1, 0, 0], scene.centroid)
                
                # Зберігаємо 3D візуалізацію
                png = scene.save_image(resolution=[800, 600])
                
                with open('object_3d_render.png', 'wb') as f:
                    f.write(png)
                    
                print("3D рендер збережено: object_3d_render.png")
        except Exception as e:
            print(f"Не вдалося створити 3D-візуалізацію: {e}")

def main():
    parser = argparse.ArgumentParser(description='Покращений консольний додаток для 3D-реконструкції без фону')
    parser.add_argument('--image', '-i', required=True, help='Шлях до вхідного зображення')
    parser.add_argument('--output', '-o', default='3d_model.obj', help='Шлях для збереження 3D-моделі')
    parser.add_argument('--resolution', '-r', nargs=2, type=int, default=[256, 256], 
                        help='Розширення 3D-моделі (висота ширина)')
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Пристрій для обчислень (cuda або cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Файл {args.image} не знайдено")
        return
    
    try:
        model = ImageTo3D(device=args.device)
        model.process_and_save(args.image, args.output)
        print(f"Готово: {args.output}")
    except Exception as e:
        print(f"Помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()