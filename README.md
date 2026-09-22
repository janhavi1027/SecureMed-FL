#  SecureMed-FL — Privacy-Preserving Federated Learning for Medical Imaging

> Training AI on hospital data **without ever moving the data.**
> A decentralized deep learning system that lets multiple hospitals collaboratively train a shared medical image reconstruction model — while every patient scan stays locally on-site, fully compliant with HIPAA/GDPR.

---

##  The Problem

Medical AI is stuck in a privacy bottleneck: the best models need data from *many* hospitals to generalize well, but hospitals legally and ethically **cannot** pool raw patient scans into one central server. Most institutions end up training weak, siloed models on small local datasets instead.

##  The Solution

**SecureMed-FL** flips the traditional ML pipeline with **Federated Learning**: *"Bring the code to the data, not the data to the code."*

Instead of centralizing images, each hospital trains an **unsupervised AutoEncoder** locally on its own scans. Only the *mathematical model weights* — never a single pixel of patient data — are sent to a central server, which aggregates them using the **FedAvg** algorithm into one stronger global model. That global model is sent back to every hospital, and the cycle repeats.

**Result:** all the accuracy gains of a large, diverse dataset — with zero data ever leaving its source.

---

##  Results — Federation Actually Works

| Model Type | Avg. MSE Loss | Reconstruction Quality | Privacy Risk |
|---|---|---|---|
| Standalone Local Model | 0.045 – 0.060 | Blurry, missing structural detail | None (but weak model) |
| **Federated Global Model** | **0.008 – 0.015** | **Sharp, structurally accurate** | **Zero — data never leaves source** |

The federated model achieves a **~3–5x reduction in reconstruction error** over any single hospital's standalone model, without a single image ever crossing a network boundary.

---

##  System Architecture

```
 Hospital A                Hospital B                Hospital C
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ Local Scans  │         │ Local Scans  │         │ Local Scans  │
│ (never leave)│         │ (never leave)│         │ (never leave)│
│      │       │         │      │       │         │      │       │
│  AutoEncoder │         │  AutoEncoder │         │  AutoEncoder │
│  (client.py) │         │  (client.py) │         │  (client.py) │
└──────┬───────┘         └──────┬───────┘         └──────┬───────┘
       │  weights only          │  weights only          │  weights only
       │  (gRPC, encrypted)     │                        │
       └───────────┬────────────┴────────────┬───────────┘
                    ▼                         
          ┌───────────────────┐
          │   FedAvg Server    │
          │   (server.py)      │
          │  aggregates weights│
          └─────────┬──────────┘
                     │  updated global model
                     ▼
        Broadcast back to all hospitals
              (repeat each round)
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Core ML Engine | Python, PyTorch |
| Federated Learning Framework | [Flower (FLWR)](https://flower.ai/) |
| Dataset | MedMNIST v2 |
| Communication | gRPC (encrypted weight transport) |
| Aggregation Algorithm | FedAvg |
| Analysis & Visualization | NumPy, Matplotlib, Torchvision |

---

##  Project Structure

```
SecureMed-FL/
├── clients/
│   └── client.py           # Local hospital training loop (MSE loss)
├── server/
│   └── server.py           # Central FedAvg aggregation coordinator
├── global_model/
│   └── model.py            # AutoEncoder architecture definition
├── setup_data.py           # Downloads & partitions MedMNIST into hospital silos
├── visualization_results.py# Benchmarks global vs. standalone models
└── requirements.txt
```

---

##  How It Works — Step by Step

1. **`setup_data.py`** downloads the MedMNIST dataset and splits it into isolated "hospital" folders, simulating real-world data silos.
2. Each **hospital client** (`clients/client.py`) trains an AutoEncoder locally to minimize MSE reconstruction loss on its own scans only.
3. Clients send **only their model weights** to the central server over gRPC — raw images never leave the client.
4. The **server** (`server/server.py`) runs FedAvg, averaging weights across all connected hospitals into one improved global model.
5. The updated global model is broadcast back to every client, and the process repeats for multiple rounds.
6. **`visualization_results.py`** benchmarks the federated global model against standalone local models and renders reconstructed scans side-by-side.

---

##  Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/janhavi1027/SecureMed-FL.git
cd SecureMed-FL
pip install -r requirements.txt
```

### 2. Prepare Data Silos
```bash
python setup_data.py
```

### 3. Run the Simulation
Open three terminals to simulate a server and two hospital clients concurrently:
```bash
# Terminal 1 — Server
python -m server.server

# Terminal 2 — Hospital Client 1
python -m clients.client

# Terminal 3 — Hospital Client 2
python -m clients.client
```

### 4. Visualize Results
```bash
python visualization_results.py
```

> **Note:** Due to GitHub file size limits, the large dataset/evaluation folders (`uploads/`, `server_test_data/`) are hosted separately on [Google Drive](https://drive.google.com/drive/folders/1KflYhnzGMdp07ICxYE55Y-k6UT5xHUO6). Download and extract them into the project root before running the simulation.

---

## 🔒 Security & Compliance

- **Zero-Data Transfer** — Raw biomedical images never leave the hospital that owns them.
- **gRPC-Encrypted Channels** — Only abstract weight tensors are transmitted, minimizing the attack/leak surface.
- **HIPAA/GDPR-Aligned by Design** — Because patient data is never centralized, the architecture avoids the core compliance risk that centralized medical AI systems face.

---

##  Roadmap / Future Work

- [ ] Add differential privacy noise to weight updates for a second layer of defense
- [ ] Support secure aggregation (encrypted weight summation) so the server never sees individual client updates
- [ ] Containerize server/client with Docker for easier multi-machine deployment
- [ ] Extend beyond MedMNIST to real DICOM medical imaging formats
- [ ] Add a lightweight dashboard (Flask/Streamlit) to monitor training rounds in real time

*(This is a research/academic simulation — it currently runs locally to demonstrate the federated learning approach; it has not been deployed to a live multi-hospital production environment.)*

---
