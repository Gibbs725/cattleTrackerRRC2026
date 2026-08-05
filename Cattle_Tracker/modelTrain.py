from ultralytics import YOLO

#Model Training
model = YOLO('yolo26n.pt')
model.train(
    data="files/teatModel/data.yaml",
    epochs=75,
    imgsz=640
)
