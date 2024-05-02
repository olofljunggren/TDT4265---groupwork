from ultralytics import YOLO, settings, utils

# Load a model
# model = YOLO("yolov8s.pt")  # load a pretrained model
model = YOLO("runs/detect/train8/weights/best.pt")  # load best model

# Use the model
model.train(data="config.yaml", epochs=1, translate=0.1, scale=0.5, shear=0.0, perspective=0.0001, flipud=0.0, fliplr=0.5, mosaic=1.0, optimizer="SGD")  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

# res=model.predict(
#   source="data/test/images/frame_000332.PNG",
#   conf=0.60,
#   save=False
# )

# for result in res:
    
#     result.boxes.xyxy   # box with xyxy format, (N, 4)
#     result.boxes.xywh   # box with xywh format, (N, 4)
#     result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
#     result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
#     result.boxes.conf   # confidence score, (N, 1)
#     result.boxes.cls    # cls, (N, 1)

# model.metrics()

metrics = model.val(data='config.yaml')

# Predict
# results = model.predict(source='data/test/images', conf=0.60, save = False)
