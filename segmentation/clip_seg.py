import torch
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('clipseg/weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);

# load and normalize image
# image_path = '/building-mapper/pairs/pair3/view_1.png'
image_path = '/building-mapper/floor_01_partial_room_03_pano_31.jpg'
image_name = image_path.split('/')[-1].split('.')[0]
input_image = Image.open(image_path)
orig_h, orig_w = input_image.size[1], input_image.size[0]

# or load from URL...
# image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
# input_image = Image.open(requests.get(image_url, stream=True).raw)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])
img = transform(input_image).unsqueeze(0)

prompts = ['window', 'door', 'front wall', 'ceiling', 'floor']
l = len(prompts)
# predict
with torch.no_grad():
    preds = model(img.repeat(l,1,1,1), prompts)[0]

print(preds.shape)

# visualize prediction
_, ax = plt.subplots(1, l+1, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(input_image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(l)];
[ax[i+1].text(0, -15, prompts[i]) for i in range(l)];
plt.savefig("clipseg.png")


import torch
import numpy as np
import matplotlib.pyplot as plt
# assume:
#   preds:        (N, 1, H, W) raw logits from CLIPSeg (H=W=352 here)
#   input_image:  PIL image (any size)

# 1. resize & convert PIL → NumPy
input_np = np.array(input_image.resize((352, 352)))   # now (352, 352, 3), uint8

# 2. compute per-class probabilities
probs = torch.sigmoid(preds[:, 0, :, :])   # (N, 352, 352)

# 3. find max probability and argmax class at each pixel
max_probs, argmax_cls = probs.max(dim=0)   # both (352, 352)

# 4. threshold for background vs object
threshold = 0.5
labels = torch.zeros_like(argmax_cls)   # (352, 352)
mask_fg = max_probs > threshold
labels[mask_fg] = argmax_cls[mask_fg] + 1                  # classes 1…N
labels = labels.cpu().numpy()
# OpenCV wants (W, H) ordering for size
import cv2
labels_up = cv2.resize(
    labels,                 # source array
    (orig_w, orig_h),       # (width, height) of target
    interpolation=cv2.INTER_NEAREST
).astype(np.uint8)
np.save(f'{image_name}.npy', labels_up)

# cv2.imwrite("labels_origsize.png", labels_up)


# 5. define colors (0=black background)
colors = np.vstack([
    [  0,   0,   0],   # background
    [255,   0,   0],   # prompt 0
    [  0, 255,   0],   # prompt 1
    [  0,   0, 255],   # prompt 2
    [255, 255,   0],   # prompt 3
    [255,   0, 255],   # prompt 3
]).astype(np.uint8)    # shape (N+1, 3)

# 6. build the color mask
color_mask = colors[labels]   # (352, 352, 3)

# 7. overlay if desired
alpha = 0.5
overlay = (input_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

# 8. visualize
fig, axes = plt.subplots(1, 3, figsize=(12,4))
axes[0].imshow(input_np);   axes[0].axis("off");   axes[0].set_title("Input (352×352)")
axes[1].imshow(color_mask); axes[1].axis("off");   axes[1].set_title("Merged Mask")
axes[2].imshow(overlay);    axes[2].axis("off");   axes[2].set_title("Overlay")
plt.tight_layout()
cv2.imwrite("binary_mask.png", (color_mask).astype(np.uint8))
plt.savefig("prediction_res.png")