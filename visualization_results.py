import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from global_model.model import GlobalAutoEncoder

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_MODEL_PATH = "global_model/global_model.pth"
TEST_DATA_DIR = "server_test_data"
SAVE_DIR = "comparison_results"  # Is folder mein save hoga

# Folder banane ka logic
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created folder: {SAVE_DIR}")

def load_test_data(folder_path):
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: {folder_path} directory not found!")
        return None
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
    if not files:
        print("Error: No PNG images found in test directory!")
        return None

    for filename in files:
        img = Image.open(os.path.join(folder_path, filename)).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).astype(np.float32) / 255.0
        images.append(img_array)
    
    return torch.tensor(np.array(images)).unsqueeze(1).to(DEVICE)

def get_loss(model, data_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            img = batch[0]
            outputs = model(img)
            loss = criterion(outputs, img)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def run_visualization():
    test_tensors = load_test_data(TEST_DATA_DIR)
    if test_tensors is None: return
    test_loader = DataLoader(TensorDataset(test_tensors), batch_size=32)

    # Models Setup
    global_model = GlobalAutoEncoder().to(DEVICE)
    if os.path.exists(GLOBAL_MODEL_PATH):
        global_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH))
    
    local_model = GlobalAutoEncoder().to(DEVICE) # Fresh/Untrained model

    # Calculation
    global_loss = get_loss(global_model, test_loader)
    local_loss = get_loss(local_model, test_loader)
    global_acc = max(0, 100 * (1 - global_loss))
    local_acc = max(0, 100 * (1 - local_loss))

    # Plotting
    labels = ['Local (Untrained)', 'Global (Federated)']
    losses = [local_loss, global_loss]
    accuracies = [local_acc, global_acc]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss Bar
    ax1.set_xlabel('Models')
    ax1.set_ylabel('MSE Loss (Lower is Better)', color='tab:red')
    bars = ax1.bar(labels, losses, color=['#ff9999', '#66b3ff'], alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Accuracy Line
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy % (Higher is Better)', color='tab:green')
    ax2.plot(labels, accuracies, color='tab:green', marker='o', linewidth=3)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('Final Project Comparison: Local vs Global Model')
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 5), ha='center', va='bottom')

    fig.tight_layout()

    # --- YAHAN SAVE HO RAHA HAI FOLDER KE ANDAR ---
    save_path = os.path.join(SAVE_DIR, "final_comparison_report.png")
    plt.savefig(save_path)
    
    print("\n" + "="*40)
    print(f"Results saved in folder: {SAVE_DIR}")
    print(f"Image Name: final_comparison_report.png")
    print(f"Global Accuracy: {global_acc:.2f}%")
    print("="*40)
    
    plt.show()

if __name__ == "__main__":
    run_visualization()