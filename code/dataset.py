from PIL import Image
import torch
from torch.utils.data import Dataset
import json

class MyDataSet_d(Dataset):
    """Dataset used by AMT-JRD.

    The label order in ``three_JRD_info.json`` is KPD, OD, and IS. The three
    attributes fed to AFFM are read from ``object_attributes.json``.
    """

    def __init__(self, images_paths, images_names, JRD_info_dict,
                 object_attributes_path, transform=None):
        self.images_paths = images_paths
        self.images_names = images_names
        self.JRD_info_dict = JRD_info_dict
        self.transform = transform
        with open(object_attributes_path, "r", encoding="utf-8") as f:
            self.object_attributes = json.load(f)

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        path = self.images_paths[item]
        img = Image.open(path).convert("RGB")
        name = self.images_names[item]
        JRD_label = self.JRD_info_dict[name]

        attributes = self.object_attributes[name]
        attributes = [attributes[2], attributes[4], attributes[5]]

        if self.transform is not None:
            img = self.transform(img)
        return img, name, JRD_label, attributes

    @staticmethod
    def collate_fn(batch):
        images, names, JRD_labels, attributes = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        JRD_labels = torch.as_tensor(JRD_labels, dtype=torch.long)
        attributes = torch.as_tensor(attributes, dtype=torch.float32)
        return images, names, JRD_labels, attributes
