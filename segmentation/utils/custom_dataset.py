
import os
import cv2
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

    
class Dataset(Dataset):
    def __init__(self, domain_name=None, base_dir=None, leng=10):
        self.base_dir = base_dir
        self.domain_name = domain_name
        self.leng = leng
        self.id_path = [x for x in os.listdir(os.path.join(self.base_dir, self.domain_name, 'images'))
                        if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"total {len(self.id_path)} samples")

    def __len__(self):
        return len(self.id_path)

    def __getitem__(self, index):
        id = self.id_path[index]
        # print(id)
        img = cv2.imread(os.path.join(self.base_dir, self.domain_name, 'images', id))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img[..., ::-1] ## RGB to BGR
        image = np.ascontiguousarray(image)
        
        mask = cv2.imread(os.path.join(self.base_dir, self.domain_name, 'masks', id), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        gt = np.array(mask)
        return image,gt,id

    def pad_to_divisible_by_32(self, img, mask):
        height, width = img.shape[:2]
        target_height = ((height + 31) // 32) * 32
        target_width = ((width + 31) // 32) * 32

        padded_img = np.zeros((target_height, target_width, img.shape[2]), dtype=img.dtype)
        padded_img[:height, :width] = img

        padded_mask = np.zeros((target_height, target_width), dtype=mask.dtype)
        padded_mask[:height, :width] = mask

        return padded_img, padded_mask
    
    @staticmethod
    def custom_collate(batch):
        images = []
        seg_labels = []
        images_name = []
        for image, gt, image_name in batch:
            images.append(image)
            seg_labels.append(gt)
            images_name.append(image_name)
        images = np.array(images)
        seg_labels = np.array(seg_labels)
        return images, seg_labels, images_name