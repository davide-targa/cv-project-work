import torch
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes

from utils.datasets import PennFudanDataset, PennFudanTextDataset

pf = PennFudanDataset()[0]
pft = PennFudanTextDataset()

output_image = draw_bounding_boxes(pf[0], boxes=pf[1]["boxes"], colors="red", width=2)
# plt.figure(figsize=(12, 12))
# plt.imshow(output_image.permute(1, 2, 0))
# plt.show()
to_pil_image(output_image).save("output_image.png")
