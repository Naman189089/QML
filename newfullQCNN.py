import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. Dataset Loading and Preprocessing
# ==========================================
def prepare_mnist_dataloaders(batch_size: int = 16, max_train_samples: int = 60000, max_test_samples: int = 200):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(max_train_samples))
    test_subset = Subset(test_dataset, range(max_test_samples))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Device & Layer Definitions
# ==========================================
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def full_zz_entanglement_layer(rot_weights, zz_weights, wires):
    """
    Implements U_full = O_ZZ^FullEnt * (⊗ Rot(alpha, beta, gamma))
    Connects every qubit in the layer with all-to-all ZZ gates.
    """
    n = len(wires)
    
    # 1. Single qubit rotations on all wires
    for i in range(n):
        qml.Rot(rot_weights[i, 0], rot_weights[i, 1], rot_weights[i, 2], wires=wires[i])
        
    # 2. All-to-all ZZ interactions: exactly n(n-1)/2 entangling gates
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ(zz_weights[idx], wires=[wires[i], wires[j]])
            idx += 1

def quantum_pooling_layer(pool_weights, source_wires, target_wires):
    """
    Uses controlled rotations to pool information from source wires into target wires.
    """
    for i in range(len(source_wires)):
        qml.CRot(pool_weights[i, 0], pool_weights[i, 1], pool_weights[i, 2], 
                 wires=[source_wires[i], target_wires[i]])

# ==========================================
# 3. Pure Quantum QNode Definition
# ==========================================
@qml.qnode(dev, interface="torch")
def full_zz_qcnn(inputs, conv1_rot, conv1_zz, pool_w, conv2_rot, conv2_zz):
    # Amplitude Encoding for 16x16 image (256 pixels -> 8 qubits)
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    
    # --- Layer 1: Full ZZ Entangled Convolution (8 Qubits) ---
    full_zz_entanglement_layer(conv1_rot, conv1_zz, wires=range(8))
    
    # --- Layer 2: Quantum Pooling (8 -> 4 Qubits) ---
    quantum_pooling_layer(pool_w, source_wires=[1, 3, 5, 7], target_wires=[0, 2, 4, 6])
    
    # --- Layer 3: Full ZZ Entangled FC/Conv (4 Remaining Qubits) ---
    full_zz_entanglement_layer(conv2_rot, conv2_zz, wires=[0, 2, 4, 6])
    
    # --- Readout: Probabilities over 16 basis states ---
    return qml.probs(wires=[0, 2, 4, 6])

# ==========================================
# 4. PyTorch Neural Network Module
# ==========================================
class FullZZQuantumMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 1: 8 qubits -> Rotations (8, 3), ZZ gates = 8*(8-1)/2 = 28
        self.conv1_rot = nn.Parameter(torch.rand(8, 3, requires_grad=True))
        self.conv1_zz = nn.Parameter(torch.rand(28, requires_grad=True))
        
        # Pooling: 4 control-target pairs -> CRotations (4, 3)
        self.pool_w = nn.Parameter(torch.rand(4, 3, requires_grad=True))
        
        # Layer 2: 4 qubits -> Rotations (4, 3), ZZ gates = 4*(4-1)/2 = 6
        self.conv2_rot = nn.Parameter(torch.rand(4, 3, requires_grad=True))
        self.conv2_zz = nn.Parameter(torch.rand(6, requires_grad=True))

    def forward(self, x):
        batch_size = x.shape[0]
        batch_log_probs = []
        
        for i in range(batch_size):
            quantum_probs = full_zz_qcnn(
                x[i], 
                self.conv1_rot, self.conv1_zz, 
                self.pool_w, 
                self.conv2_rot, self.conv2_zz
            )
            # Slice top 10 basis states (|0000> to |1001>) for digits 0-9
            class_probs = quantum_probs[:10]
            
            # Re-normalize so sum over 10 classes equals 1.0
            sum_probs = torch.sum(class_probs)
            normalized_probs = class_probs / (sum_probs + 1e-9)
            
            # Compute log-probabilities for NLLLoss
            log_probs = torch.log(normalized_probs + 1e-9)
            batch_log_probs.append(log_probs)
            
        return torch.stack(batch_log_probs)

# ==========================================
# 5. Training and Evaluation Pipeline
# ==========================================
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")

def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, dim=1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            
    accuracy = (correct_predictions / total_samples) * 100.0
    print("=====================================")
    print(f"Full ZZ QCNN Clean Test Accuracy: {accuracy:.2f}%")
    print("=====================================")
    return accuracy

# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 15
    LEARNING_RATE = 0.01  # Reduced learning rate for gradient stability

    print("Preparing 16x16 MNIST Dataloaders...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    print("Initializing Full ZZ Quantum Model...")
    model = FullZZQuantumMNIST()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Total Trainable Quantum Parameters: {sum(p.numel() for p in model.parameters())}")

    print("Starting Training Loop...")
    train_model(model, train_loader, optimizer, criterion, EPOCHS)

    print("Starting Evaluation Loop...")
    evaluate_model(model, test_loader)