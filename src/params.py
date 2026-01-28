from models.fasterrcnn_resnet50 import get_model as resnet_model
from models.fasterrcnn_resnet50_v2 import get_model as resnet_v2_model
from models.ssd300_vgg16 import get_model as ssd300_model

model = resnet_model()
print(f"Total parameters fasterrcnn_resnet50: {sum(p.numel() for p in model.parameters())}")

model = resnet_v2_model()
print(f"Total parameters fasterrcnn_resnet50_v2: {sum(p.numel() for p in model.parameters())}")

model = ssd300_model()
print(f"Total parameters ssd300_vgg16: {sum(p.numel() for p in model.parameters())}")
