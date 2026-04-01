import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/sym1502/下载/D2E2-Det/runs/train/DroneVehicle/COD+DCE-yolov11s/weights/best.pt') # select your model.pt path
    model.predict(source='/home/sym1502/下载/DroneVehicle_OBB_MultiModal/images/val/01172.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='3',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=False, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )