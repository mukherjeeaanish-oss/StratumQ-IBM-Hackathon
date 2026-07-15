import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN


class QuantumMedicalHead(nn.Module):
    """STRATUM Q: CLASSICAL-QUANTUM-CLASSICAL HYBRID CLASSIFICATION HEAD (Qiskit v1.0+ V2 Spec)"""

    def __init__(self, input_features=512, num_classes=3):
        super().__init__()

        # Compress 512 ResNet features down to 2 numbers
        self.classical_compressor = nn.Linear(input_features, 2)

        self.qc = QuantumCircuit(2)
        # ⚡ FIXED: Feature Maps require 'feature_dimension', Ansatzes require 'num_qubits'
        self.feature_map = ZZFeatureMap(feature_dimension=2, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=2, reps=1)

        self.qc.compose(self.feature_map, inplace=True)
        self.qc.compose(self.ansatz, inplace=True)

        self.qnn = SamplerQNN(
            sampler=StatevectorSampler(),
            circuit=self.qc,
            input_params=self.feature_map.parameters,
            weight_params=self.ansatz.parameters,
        )

        self.quantum_layer = TorchConnector(self.qnn)
        self.classical_decoder = nn.Linear(2**2, num_classes)

    def forward(self, x):
        x_compressed = self.classical_compressor(x)
        quantum_state_probs = self.quantum_layer(x_compressed)
        logits = self.classical_decoder(quantum_state_probs)
        return logits