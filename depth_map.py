import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import filters

class DepthMap:
    def __init__(self, model_type="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model = None
        self.transform = None

    def load_model(self):
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = transforms.dpt_transform if "DPT" in self.model_type else transforms.small_transform
            print(f"Модель {self.model_type} завантажена")
        except:
            print("Не вийшло завантажити модель, будемо рахувати глибину простим методом")
            self.model = "simple"

    def enhance_depth(self, depth, img):
        """Комплексне покращення карти глибини"""
        # Підвищення контрасту
        depth = cv2.equalizeHist((depth * 255).astype(np.uint8)) / 255.0
        depth = filters.gaussian(depth, sigma=1.0)
        
        # Додавання країв
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        edges = filters.sobel(gray)
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        depth = depth + 0.3 * edges
        
        # Нормалізація та гамма корекція
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.power(depth, 0.7)
        
        # Посилення центру
        h, w = depth.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - w//2)**2 + (y - h//2)**2)
        center_mask = 1 - (dist / np.sqrt((w//2)**2 + (h//2)**2))
        depth = depth * (1 + 0.5 * center_mask)
        
        # Сегментація об'єкта
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(mask, [max(contours, key=cv2.contourArea)], 255)
            depth = np.where(mask > 0, depth * (mask/255.0) + 0.3, depth * 0.1)
        
        return (depth - depth.min()) / (depth.max() - depth.min())

    def estimate_depth(self, img):
        """Оцінка глибини"""
        if self.model == "simple":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(grad_x**2 + grad_y**2)
            laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            
            depth = (edges / (edges.max() + 1e-8)) * 0.4 + \
                    (laplacian / (laplacian.max() + 1e-8)) * 0.3 + \
                    ((255 - blur) / 255.0) * 0.3
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img_rgb).to(self.device)
            
            with torch.no_grad():
                pred = self.model(input_batch)
                pred = F.interpolate(pred.unsqueeze(1), size=img_rgb.shape[:2], 
                                   mode="bicubic", align_corners=False).squeeze()
            
            depth = pred.cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return self.enhance_depth(depth, img)

    def show_depth(self, original, depth, model_type):
        """Візуалізація результатів"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Оригінал")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(depth, cmap='plasma')
        plt.title(f"Карта глибин {model_type}")
        plt.axis('off')
        
        plt.savefig(f"depth_map_{model_type}.png", bbox_inches='tight', dpi=150) 
        print(f"Збережено як depth_map_{model_type}.png")

def main():
    img_path = "cat.jpg"
    models = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    
    # Завантаження зображення
    img = cv2.imread(img_path)
    if img is None:
        print("Не знайдено зображення, створюю випадкове...")
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite("test.jpg", img)
        img = cv2.imread("test.jpg")

    # Обробка всіх моделей
    for model_type in models:
        dpt = DepthMap(model_type)
        dpt.load_model()
        depth = dpt.estimate_depth(img)
        dpt.show_depth(img, depth, model_type)

if __name__ == "__main__":
    main()