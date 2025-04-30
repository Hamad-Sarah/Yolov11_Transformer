import torch
from ultralytics.nn.tasks import DetectionModel

# Load the modified YOLOv11n model (assumes yolo11.yaml has been updated)
model = DetectionModel(cfg="ultralytics/cfg/models/11/yolo11.yaml", ch=3)
model.eval()

# Dummy input
x = torch.randn(1, 3, 512, 512)

# Forward through backbone to print feature map shapes
y = x
for i, layer in enumerate(model.model):
    y = layer(y)
    # Print shapes at the expected P3, P4, P5 layers
    if i in [5, 9, 12]:  # indices of MobileViT output layers (see YAML above)
        print(f"Output of layer {i}: {y.shape}")
