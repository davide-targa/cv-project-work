from losses_data import fasterrcnn_resnet50_losses, ssd300_vgg16_losses
from matplotlib import pyplot as plt

plt.plot(fasterrcnn_resnet50_losses, label="Faster R-CNN ResNet50")
plt.plot(ssd300_vgg16_losses, label="SSD300 VGG16")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
