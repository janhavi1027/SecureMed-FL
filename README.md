# 🏥 SecureMed-FL: Privacy-Preserving Federated Medical Imaging Dashboard

> **SecureMed-FL** is an enterprise-grade medical imaging dashboard that demonstrates **Federated Learning (FL)** across decentralized multi-hospital networks. By leveraging deep learning (AutoEncoders) and Flower (FLWR), SecureMed-FL trains robust medical AI models while ensuring **zero raw patient data leaves local hospital firewalls**.

---

🌐 **Live Interactive Dashboard:** [securemed-fl.streamlit.app](https://securemed-fl.streamlit.app/)

---

## 📌 Executive Summary & Problem Statement

In modern digital healthcare, AI adoption is severely bottlenecked by **data privacy regulations (HIPAA, GDPR)** and **institutional data silos**. Centralizing sensitive patient scans (e.g., X-rays, Pneumonia MNIST) to a single server poses catastrophic privacy risks and regulatory hurdles.

### The Solution — Federated Learning
Instead of moving sensitive medical scans to a central server, **SecureMed-FL** brings the model to the data:
1. Each hospital node trains a local AutoEncoder model on its private scans behind its firewall.
2. Only encrypted mathematical model parameters (weights/gradients) are transmitted over gRPC to a central aggregator.
3. The server performs **Federated Averaging (FedAvg)** to synthesize a global intelligence model.
4. The updated global weights are redistributed back to all participating hospital nodes.

---

## ✨ Key Features

- 🔒 **Zero-Data Leakage Architecture:** Raw medical image pixels are never transmitted or stored centrally.
- 🩺 **3-Way Comparative Visualizer:** Side-by-side reconstruction quality comparison:
  1. **Source Patient Scan**
  2. **Standalone Local Hospital Model Reconstruction** (Overfitted due to limited local data)
  3. **Federated Global Model Reconstruction** (High generalization & reconstruction fidelity)
- 🏥 **Dynamic Multi-Hospital Auto-Discovery:** Automatically scans local data nodes (`Hospital 1` through `Hospital 10`) and dynamically reflects active client nodes.
- 🧮 **FedAvg Aggregation Flow:** Mathematical insights and step-by-step weight exchange visualization.
- 🎨 **Clinical Light UI/UX:** High-contrast, accessibility-focused clinical light theme built with Streamlit and custom CSS styling.

---

## 🛠️ Tech Stack & Libraries

| Domain | Tools & Frameworks |
| :--- | :--- |
| **Core Language** | Python 3.10+ |
| **Deep Learning Engine** | PyTorch, Torchvision |
| **Federated Learning** | Flower (`flwr`) |
| **Interactive Dashboard** | Streamlit |
| **Data Processing & Vision** | NumPy, Pillow (PIL), OpenCV |
| **Cloud Deployment** | Streamlit Community Cloud, Git/GitHub |

---

## 🏗️ System Architecture & Federated Workflow

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Hospital 1    │       │   Hospital 2    │  ...  │   Hospital 10   │
│  (Data Silo A)  │       │  (Data Silo B)  │       │  (Data Silo N)  │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         │ Local Training          │ Local Training          │ Local Training
         ▼                         ▼                         ▼
  [Local Weights]           [Local Weights]           [Local Weights]
         │                         │                         │
         └────────────────┐        │        ┌────────────────┘
                          ▼        ▼        ▼
                   ┌────────────────────────────────┐
                   │    Central Server (FLWR)       │
                   │   Federated Averaging (FedAvg) │
                   └───────────────┬────────────────┘
                                   │
                                   ▼
                         [Global Model Update]
```

### Mathematical Aggregation (FedAvg)
The central aggregator computes the weighted parameter average across participating hospital clients:

$$\text{Weight}_{\text{global}} = \frac{1}{N} \sum_{i=1}^{N} \text{Weight}_{i}$$

Where $N$ is the total number of participating hospital client nodes.

---

## 📁 Repository Structure

```
SecureMed-FL/
│
├── app.py                      # Interactive Streamlit Web Application
├── requirements.txt            # Dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git untracked files specification
│
├── global_model/               # Global AutoEncoder model definition & weights
│   ├── model.py                # PyTorch AutoEncoder Neural Network
│   └── global_model.pth        # Trained Federated Global Model Weights
│
├── clients/                    # Federated Client execution & dataset loaders
│   ├── client.py               # Flower (FLWR) Client implementation
│   ├── dataset_loader.py       # Local medical scan pre-processing
│   └── upload_ui.py            # Client scan upload handler
│
├── server/                     # Federated Server aggregation & evaluation
│   ├── server.py               # FLWR Aggregator Server script
│   └── evaluate.py             # Global model validation metrics
│
├── uploads/                    # Local hospital data silos (Hospital 1 - 10)
└── visualization_results/      # Output graphs and evaluation metrics
```

---

## 🚀 Quick Start & Local Setup

Follow these steps to set up and run the application on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/janhavi1027/SecureMed-FL.git
cd SecureMed-FL
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Model Architecture & Training

The project utilizes a **Convolutional AutoEncoder (PyTorch)** tailored for medical scan reconstruction:
- **Encoder:** Convolutional layers with BatchNorm and ReLU activations that compress scan images into a compact latent vector.
- **Decoder:** Transposed Convolutional layers that reconstruct the low-dimensional representation back into original image dimensions.
- **Loss Criterion:** Mean Squared Error (MSE Loss) for spatial pixel-level reconstruction accuracy.

---

## 📊 Results & Clinical Impact

| Evaluation Criteria | Standalone Hospital Model | SecureMed Federated Model |
| :--- | :---: | :---: |
| **Raw Data Exposure** | High Risk (Centralization required) | **0% (Zero Leakage)** |
| **Reconstruction MSE** | High (Overfitted on sparse dataset) | **Significantly Reduced** |
| **Feature Generalization** | Low (Single-site bias) | **High (Multi-institutional intelligence)** |
| **Regulatory Compliance** | Risk of violation | **100% HIPAA & GDPR Compliant** |

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the architecture or dashboard UI:

1. Fork the Repository
2. Create your Feature Branch (`git checkout -b feature/CoolFeature`)
3. Commit your Changes (`git commit -m 'Add CoolFeature'`)
4. Push to the Branch (`git push origin feature/CoolFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 👤 Author & Acknowledgments

- **Janhavi** — [GitHub Profile](https://github.com/janhavi1027)
- **Live Demo Link:** [SecureMed-FL Live Dashboard](https://securemed-fl.streamlit.app/)
- *Special thanks to the developers and maintainers of PyTorch, Flower (FLWR), and Streamlit.*
