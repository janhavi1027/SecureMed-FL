import os
import shutil
import random
import medmnist
from medmnist import INFO
from PIL import Image
from tqdm import tqdm

def setup_flexible_federated_data():
    # --- CONFIGURATION ---
    datasets = ['chestmnist', 'dermamnist', 'bloodmnist', 'pathmnist', 'octmnist', 'pneumoniamnist', 'breastmnist']
    root_dir = "uploads"
    test_dir = "server_test_data"
    
    # Speed control: Images per hospital
    MAX_TRAIN_PER_HOSPITAL = 10000
    MAX_TEST_TOTAL = 9000       
    # ---------------------

    if os.path.exists(root_dir): shutil.rmtree(root_dir)
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # AB YEH LINE RANDOM HOSPTIALS GENERATE KAREGI (5 TO 10)
    num_hospitals = random.randint(5, 10) 
    print(f"Randomly generating data for {num_hospitals} hospitals...")

    hospital_dirs = []
    for i in range(num_hospitals):
        h_dir = os.path.join(root_dir, f"Hospital_{i+1}")
        os.makedirs(h_dir)
        hospital_dirs.append(h_dir)

    per_ds_train = MAX_TRAIN_PER_HOSPITAL // len(datasets)
    per_ds_test = MAX_TEST_TOTAL // len(datasets)

    for ds_name in datasets:
        print(f" Processing {ds_name}...")
        info = INFO[ds_name]
        DataClass = getattr(medmnist, info['python_class'])
        
        train_dataset = DataClass(split='train', download=True, size=28)
        test_dataset = DataClass(split='test', download=True, size=28)

        # Distribute Training Data
        for h_dir in hospital_dirs:
            indices = random.sample(range(len(train_dataset)), min(per_ds_train, len(train_dataset)))
            for idx in indices:
                img, _ = train_dataset[idx]
                img.save(os.path.join(h_dir, f"{ds_name}_{idx}.png"))

        # Add to Global Test Set
        test_indices = random.sample(range(len(test_dataset)), min(per_ds_test, len(test_dataset)))
        for idx in test_indices:
            img, _ = test_dataset[idx]
            img.save(os.path.join(test_dir, f"{ds_name}_test_{idx}.png"))

    print("\n--- FINAL DATA DISTRIBUTION ---")
    for h in hospital_dirs:
        print(f"{os.path.basename(h)}: {len(os.listdir(h))} images")
    print(f" {test_dir}: {len(os.listdir(test_dir))} images")

if __name__ == "__main__":
    setup_flexible_federated_data()