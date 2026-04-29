import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import argparse
from tqdm import tqdm


class CoalImageLightingProcessor:
    """煤矸石图片光照变化处理器"""

    def __init__(self, input_dir, output_dir, seed=None):
        """
        初始化处理器
        参数:
            input_dir: 原始图片所在目录
            output_dir: 处理后图片保存目录
            seed: 随机种子，用于复现结果
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 支持的图片格式
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 定义 5 档固定亮度（亮 → 暗）
        self.brightness_levels = [
            (1.8, "bright"),  # 亮
            (1.3, "light"),  # 偏亮
            (1.0, "normal"),  # 正常
            (0.6, "dark"),  # 偏暗
            (0.3, "very_dark")  # 极暗
        ]

    def _adjust_brightness_with_enhance(self, img, bright_factor):
        """统一调整亮度 + 轻微自然增强"""
        # 亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(bright_factor)

        # 暗图自动加一点对比度，避免死黑
        if bright_factor < 0.7:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.15)

        # 亮图轻微降噪
        if bright_factor > 1.5:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(0.9)

        return img

    def process_all_images(self):
        """处理所有图片，生成 5 档固定光照"""
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(self.supported_formats)]

        if not image_files:
            print(f"目录 {self.input_dir} 未找到图片")
            return

        print(f"发现 {len(image_files)} 张图片，将为每张生成 5 档光照（亮→暗）")

        for img_file in tqdm(image_files, desc="处理进度"):
            try:
                img_path = os.path.join(self.input_dir, img_file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    base_name, ext = os.path.splitext(img_file)

                    # 生成 5 档固定亮度
                    for factor, level_name in self.brightness_levels:
                        result_img = self._adjust_brightness_with_enhance(img, factor)
                        output_filename = f"{base_name}_{level_name}{ext}"
                        output_path = os.path.join(self.output_dir, output_filename)
                        result_img.save(output_path, quality=95)

            except Exception as e:
                print(f"\n处理失败 {img_file}: {str(e)}")
                continue

        print(f"\n处理完成！5 档光照图片已保存到：{self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='煤矸石图片 5 档固定光照生成')
    parser.add_argument('--input', default='D:\\图片\\1', help='输入图片目录')
    parser.add_argument('--output', default='D:\\图片\\2', help='输出目录')
    parser.add_argument('--seed', type=int, help='随机种子')

    args = parser.parse_args()

    processor = CoalImageLightingProcessor(
        input_dir=args.input,
        output_dir=args.output,
        seed=args.seed
    )
    processor.process_all_images()
