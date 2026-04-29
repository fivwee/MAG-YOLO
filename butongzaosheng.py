import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm


class GaussianNoiseGenerator:
    """给图片添加不同强度的高斯噪声"""

    def __init__(self, input_dir, output_dir, seed=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

        if seed is not None:
            np.random.seed(seed)

        # 定义 4 档高斯噪声强度（数字越大噪声越强）
        self.noise_levels = [
            (5, "weak"),      # 弱噪声
            (12, "medium"),    # 中等噪声
            (25, "strong"),    # 强噪声
            (40, "extreme")    # 极强噪声
        ]

    def add_gaussian_noise(self, img, sigma):
        """
        给图片添加高斯噪声
        :param img: PIL Image
        :param sigma: 噪声强度
        :return: 加噪后的图片
        """
        img_np = np.array(img).astype(np.float32)

        # 生成高斯噪声
        noise = np.random.normal(loc=0, scale=sigma, size=img_np.shape)

        # 添加噪声并裁剪到 0-255
        img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_noisy)

    def process_all_images(self):
        # 获取图片列表
        image_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(self.supported_formats)]
        if not image_files:
            print(f"未在 {self.input_dir} 找到图片")
            return

        print(f"找到 {len(image_files)} 张图片，开始添加高斯噪声...")

        for img_file in tqdm(image_files, desc="生成噪声图片"):
            try:
                img_path = os.path.join(self.input_dir, img_file)
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    base_name, ext = os.path.splitext(img_file)

                    # 为每张图生成 4 种强度的高斯噪声
                    for sigma, level_name in self.noise_levels:
                        noisy_img = self.add_gaussian_noise(img, sigma)
                        save_name = f"{base_name}_gaussian_{level_name}{ext}"
                        save_path = os.path.join(self.output_dir, save_name)
                        noisy_img.save(save_path, quality=95)

            except Exception as e:
                print(f"\n处理失败 {img_file}: {str(e)}")
                continue

        print(f"\n✅ 高斯噪声生成完成！保存至：{self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='给图片添加高斯噪声')
    parser.add_argument('--input', default='D:\\图片\\1', help='输入图片目录')
    parser.add_argument('--output', default='D:\\图片\\gaussian_noise', help='输出目录')
    parser.add_argument('--seed', type=int, help='随机种子')

    args = parser.parse_args()

    generator = GaussianNoiseGenerator(
        input_dir=args.input,
        output_dir=args.output,
        seed=args.seed
    )
    generator.process_all_images()

