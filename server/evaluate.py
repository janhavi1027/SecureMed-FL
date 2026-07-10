import torch
import os
import math
from PIL import Image
import torchvision.transforms as transforms
from flwr.common import parameters_to_ndarrays
from global_model.model import GlobalAutoEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_global_model(parameters):
    model = GlobalAutoEncoder().to(DEVICE)
    ndarrays = parameters_to_ndarrays(parameters)
    state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(BASE_DIR, "server_test_data")

    image_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        return 1.0, 0.0

    for img_path in image_files:
        img = Image.open(img_path)
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            total_loss += loss_fn(out, x).item()

    avg_loss = total_loss / len(image_files)
    accuracy = max(0.0, (1.0 - math.sqrt(avg_loss)) * 100.0)
    
    return avg_loss, accuracy