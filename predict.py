import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

print("⚙️ Booting Quantum-Hybrid Engine...")

# --- CONFIGURATION ---
NUM_CLASSES = 3
WEIGHTS_PATH = 'stratum_q_resp_BEST.pth'
IMAGE_PATH = 'test_scan.jpg'  # The CT Scan we want to diagnose

CLASS_NAMES = ['Positive: Malignant', 'Negative for Cancer/Pneumonia', 'Positive: Pneumonia']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"⚡ COMPUTE ANCHORED TO: {device}")

# --- 1. REBUILD THE ARCHITECTURE ---
def create_qnn_head():
    fmap = ZZFeatureMap(feature_dimension=2, reps=1)
    ansatz = RealAmplitudes(num_qubits=2, reps=1)
    
    qc = QuantumCircuit(2)
    qc.compose(fmap, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    def interpret_func(x):
        return x % NUM_CLASSES

    return SamplerQNN(
        circuit=qc,
        input_params=fmap.parameters,     
        weight_params=ansatz.parameters,  
        interpret=interpret_func,
        output_shape=NUM_CLASSES,
        input_gradients=True              
    )

class StratumResp(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.base = models.resnet18(weights=None) 
        self.base.fc = nn.Linear(self.base.fc.in_features, 2) 
        self.qnn_conn = TorchConnector(qnn)

    def forward(self, x): 
        x_class = self.base(x)
        x_quant = self.qnn_conn(x_class)
        return x_quant

# --- 2. LOAD YOUR 97.42% WEIGHTS ---
print("📥 Loading Stratum Thesis Weights (97.42% Acc)...")
model = StratumResp(create_qnn_head()).to(device)

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=True))
model.eval() 

# --- 3. PREPARE THE CT SCAN ---
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print(f"🔬 Scanning Image: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert('RGB')
img_tensor = preprocess(img).unsqueeze(0).to(device)

# --- 4. THE LIVE PREDICTION ---
print("🚀 Firing Quantum Circuit...")
with torch.no_grad():
    raw_output = model(img_tensor)
    
    probabilities = raw_output[0] * 100
    predicted_idx = torch.argmax(raw_output, 1).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

print("\n" + "="*40)
print(" 🩺 FINAL QUANTUM DIAGNOSIS 🩺 ")
print("="*40)
print(f"🎯 Prediction : {predicted_class.upper()}")
print(f"📊 Confidence : {confidence:.2f}%\n")

print("Detailed Scan Breakdown:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f" - {class_name}: {probabilities[i].item():.2f}%")
print("="*40)