from ultralytics import YOLO

#Model Training
model = YOLO('yolo26n-pose.pt')
model.train(
    data="files/frameworkModel/dataset.yaml",
    epochs=50,
    imgsz=640
)