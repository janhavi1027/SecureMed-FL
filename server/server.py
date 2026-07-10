import flwr as fl
import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from global_model.model import GlobalAutoEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "global_model/global_model.pth"

def load_images_from_folder(folder_path):
    images = []
    if not os.path.exists(folder_path):
        return None
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename)).convert('L') # Grayscale
            img = img.resize((28, 28))
            img_array = np.array(img).astype(np.float32) / 255.0 # Normalize
            images.append(img_array)
    if not images: return None
    return torch.tensor(np.array(images)).unsqueeze(1) # Shape: [N, 1, 28, 28]

def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            img = batch[0].to(DEVICE)
            loss = criterion(model(img), img)
            total_loss += loss.item()
    avg_loss = total_loss / len(test_loader)
    accuracy = max(0, 100 * (1 - avg_loss))
    return avg_loss, accuracy

class ManualStrategy(fl.server.strategy.FedAvg):
    def __init__(self, test_loader, *args, **kwargs):
        self.test_loader = test_loader
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        agg_params, _ = super().aggregate_fit(server_round, results, failures)
        if agg_params:
            ndarrays = fl.common.parameters_to_ndarrays(agg_params)
            model = GlobalAutoEncoder().to(DEVICE)
            state_dict = zip(model.state_dict().keys(), [torch.tensor(v) for v in ndarrays])
            model.load_state_dict({k: v for k, v in state_dict})
            if not os.path.exists("global_model"): os.makedirs("global_model")
            torch.save(model.state_dict(), MODEL_PATH)
            
            loss, acc = evaluate_model(model, self.test_loader)
            print("\n" + "="*45 + f"\nROUND {server_round} TEST RESULTS\nLoss: {loss:.6f} | Accuracy: {acc:.2f}%\n" + "="*45)
        return agg_params, {}

def start_server():
    test_dir = "server_test_data"
    x_test = load_images_from_folder(test_dir)
    
    if x_test is None:
        print(f"❌ Error: No images in {test_dir}!")
        x_test = torch.rand(10, 1, 28, 28)
    
    test_loader = DataLoader(TensorDataset(x_test), batch_size=32)
    model = GlobalAutoEncoder().to(DEVICE)
    if os.path.exists(MODEL_PATH): model.load_state_dict(torch.load(MODEL_PATH))

    print(f"\n🚀 ROUND 0: INITIAL TESTING ({len(x_test)} images)")
    l, a = evaluate_model(model, test_loader)
    print(f"Initial Loss: {l:.6f} | Acc: {a:.2f}%\n" + "-"*40)

    strategy = ManualStrategy(test_loader=test_loader, min_fit_clients=1, min_available_clients=1,
                              initial_parameters=fl.common.ndarrays_to_parameters([v.cpu().numpy() for v in model.state_dict().values()]))

    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=1), strategy=strategy)

if __name__ == "__main__":
    start_server()