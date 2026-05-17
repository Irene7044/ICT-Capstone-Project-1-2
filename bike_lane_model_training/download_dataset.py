from roboflow import Roboflow

rf = Roboflow(api_key="QRkRMkjLsX5YPYcGxumT")
project = rf.workspace("tathithaominh-gmail-com").project("bicycle-lanes-segmentation-1qozn")
version = project.version(1)
dataset = version.download("yolov8")
                