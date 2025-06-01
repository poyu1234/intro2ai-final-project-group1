from data import generate_images
import os

def main():
    # 生成100張帶有文字的圖片
    num_samples = 100
    N = 4  # 每張圖片的矩形數量
    image_size = 256  # 圖片大小
    add_text = True  # 確保添加文字
    line_width_range = (3,5)  # 線條寬度範圍
    
    print(f"正在生成 {num_samples} 張帶有文字的圖片...")
    
    # 生成圖片
    original_images, noisy_images = generate_images(num_samples, N, image_size, add_text,line_thickness_range=line_width_range)
    
    # 創建保存目錄
    os.makedirs("dataset/original", exist_ok=True)
    os.makedirs("dataset/noisy", exist_ok=True)
    
    # 保存圖片
    for i, (original_img, noisy_img) in enumerate(zip(original_images, noisy_images)):
        # 保存原始圖片
        original_path = f"dataset/original/original_{i}.jpg"
        original_img.save(original_path)
        
        # 保存噪聲圖片
        noisy_path = f"dataset/noisy/noisy_{i}.jpg"
        noisy_img.save(noisy_path)
        
        if (i + 1) % 10 == 0:
            print(f"已生成 {i + 1} 張圖片...")
    
    print(f"完成！已生成 {num_samples} 張帶有文字的圖片")
    print(f"原始圖片保存在: dataset/original/")
    print(f"噪聲圖片保存在: dataset/noisy/")

if __name__ == "__main__":
    main()