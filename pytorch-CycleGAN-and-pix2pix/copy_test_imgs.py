import os
import random
from PIL import Image
from torchvision import transforms

src_dir = 'datasets/WHU-SEN-City/opt2sar/trainB'
dest_dir = 'datasets/test/sar2opt'
num_images = 5
size = (256, 256)
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

img_list = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            img_list.append(os.path.join(root, file))

selected_imgs = random.sample(img_list, min(num_images, len(img_list)))
transform = transforms.Compose([transforms.Resize(256, transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])

for i, img_path in enumerate(selected_imgs):
    img = Image.open(img_path).convert('RGB')
    img_resized = transform(img)
    img_resized = transforms.ToPILImage()(img_resized)
    img_resized.save(os.path.join(dest_dir, f"{i+1}.png"))
