import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class ElderlyBehaviorDataset(Dataset):
    def __init__(self, data_root, transform=None, pressure_len=128):
        self.image_dir = os.path.join(data_root, 'images')
        self.pressure_dir = os.path.join(data_root, 'pressure')
        self.label_dir = os.path.join(data_root, 'labels')
        self.ids = sorted(f[:-4] for f in os.listdir(self.image_dir) if f.endswith('.jpg'))
        self.transform = transform or T.Compose([
            T.Resize((480, 640)),
            T.ToTensor()
        ])
        self.pressure_len = pressure_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        
        # 加载图像
        img_path = os.path.join(self.image_dir, f"{sample_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 加载压力数据
        pressure_path = os.path.join(self.pressure_dir, f"{sample_id}.npy")
        pressure = np.load(pressure_path)
        if pressure.shape[0] > self.pressure_len:
            pressure = pressure[-self.pressure_len:]  # 截断
        else:
            pad = self.pressure_len - pressure.shape[0]
            pressure = np.pad(pressure, (pad, 0))  # 前补0
        pressure = torch.tensor(pressure, dtype=torch.float32)

        # 加载标签
        label_path = os.path.join(self.label_dir, f"{sample_id}.json")
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        label = label_data["action"]  # 如： "sit"
        
        # 将标签映射为整数
        label_map = {"sit": 0, "stand": 1, "walk": 2}
        label_id = label_map.get(label, -1)

        return {
            "image": image,
            "pressure": pressure,
            "label": label_id
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ElderlyBehaviorDataset(data_root='data/')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in dataloader:
        images = batch['image']        # shape: [B, 3, H, W]
        pressure = batch['pressure']   # shape: [B, T]
        labels = batch['label']        # shape: [B]
        break