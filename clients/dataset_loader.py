import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class UploadedDataset(Dataset):
    def __init__(self, root):
        self.images = [os.path.join(root, f) for f in os.listdir(root)]
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        x = self.transform(img)
        return x, x   # 🔥 input == target (autoencoder)


def load_client_data(dataset_path):
    dataset = UploadedDataset(dataset_path)
    return DataLoader(dataset, batch_size=16, shuffle=True)