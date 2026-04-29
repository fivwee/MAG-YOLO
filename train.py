import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:\\ultralytics-main8\\ultralytics\\models\\v3\\yolov3-tiny.yaml')
    model.train(data='D:\\ultralytics-main8\\datasets\\cg.yaml',
                cache=False, #换数据的时候更新一下
                imgsz=640,
                epochs=300,
                batch=16, #根据电脑性能选择
                # close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD',  #优化器选择  所以实 验固定一个，确定好就不能换
                # resume='YOLOv8-MG-AQM/runs-3/train/exp8/weights/last.pt', # last.pt path # 中断后是否断续
                # amp=False, # close ampfds
                # fraction=0.2,
                project='runs-3/train',
                name='yolov5-',
                amp=False, #确定是否使用自动混合精度
                iou=0.7,
                # conf=0.25, #目标检测的对象置信度阈值。只有置信度高于此阈值的对象才会被检测出来。默认值为0.25。
                # patience=10, #最后50轮没有进步 直接结束
                # seed=42, #随机程度
                close_mosaic=0, #最后10轮进行关闭数据增强
                # cos_lr=True, #是否使用余弦学习率调度器
                # resume=False,  #确定是否从上一个checkpoint继续训练模型
                # fraction=1.0,  #确定训练时要使用的数据集比例
                )