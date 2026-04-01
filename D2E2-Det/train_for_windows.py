
from multiprocessing import freeze_support
from ultralytics import YOLO
import ultralytics.nn.tasks
def main():
    model = YOLO("/home/sym1502/下载/D2E2-Det/yaml/COD+DCE-yolov11s.yaml")
    results = model.train(data='/home/sym1502/下载/D2E2-Det/data/myRGBTDronePerson.yaml',batch=1,epochs=200,imgsz=640,cache=False,
                          device=[0],
                          # resume=True,
                          project='runs/train/RGBTDronePerson',
                          # amp=False,
                          name='1')


if __name__ == "__main__":
    freeze_support()
    main()
