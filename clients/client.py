import flwr as fl
import torch
import sys
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from global_model.model import GlobalAutoEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hospital_data(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename)).convert('L')
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
    return torch.tensor(np.array(images)).unsqueeze(1)

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict}, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for _ in range(3): # 3 Local Epochs
            for batch in self.train_loader:
                img = batch[0].to(DEVICE)
                optimizer.zero_grad(); torch.nn.functional.mse_loss(self.model(img), img).backward(); optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

if __name__ == "__main__":
    hospital_folder = sys.argv[1]
    print(f"🏥 Loading PNG images from: {hospital_folder}")
    x_train = load_hospital_data(hospital_folder)
    train_loader = DataLoader(TensorDataset(x_train), batch_size=32, shuffle=True)
    
    model = GlobalAutoEncoder().to(DEVICE)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=HospitalClient(model, train_loader))