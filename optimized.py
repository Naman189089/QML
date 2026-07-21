import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. Dataset Loading and Preprocessing
# ==========================================
def prepare_mnist_dataloaders(batch_size: int = 256, max_train_samples: int = 60000, max_test_samples: int = 200):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(max_train_samples))
    test_subset = Subset(test_dataset, range(max_test_samples))

    # OPTIMIZATION: Added pin_memory and num_workers for faster CPU-to-GPU data transfer
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Device & Layer Definitions
# ==========================================
n_qubits = 8
# Using default.qubit. When combined with diff_method="backprop" and CUDA tensors, 
# this simulates purely on the GPU using PyTorch operations.
dev = qml.device("default.qubit", wires=n_qubits)

def full_zz_entanglement_layer(rot_weights, zz_weights, wires):
    n = len(wires)
    for i in range(n):
        qml.Rot(rot_weights[i, 0], rot_weights[i, 1], rot_weights[i, 2], wires=wires[i])
        
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ(zz_weights[idx], wires=[wires[i], wires[j]])
            idx += 1

def quantum_pooling_layer(pool_weights, source_wires, target_wires):
    for i in range(len(source_wires)):
        qml.CRot(pool_weights[i, 0], pool_weights[i, 1], pool_weights[i, 2], 
                 wires=[source_wires[i], target_wires[i]])

# ==========================================
# 3. Pure Quantum QNode Definition
# ==========================================
# OPTIMIZATION: diff_method="backprop" ensures PyTorch executes the simulation natively on the GPU
@qml.qnode(dev, interface="torch", diff_method="backprop")
def full_zz_qcnn(inputs, conv1_rot, conv1_zz, pool_w, conv2_rot, conv2_zz):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)
    
    full_zz_entanglement_layer(conv1_rot, conv1_zz, wires=range(8))
    quantum_pooling_layer(pool_w, source_wires=[1, 3, 5, 7], target_wires=[0, 2, 4, 6])
    full_zz_entanglement_layer(conv2_rot, conv2_zz, wires=[0, 2, 4, 6])
    
    return qml.probs(wires=[0, 2, 4, 6])

# ==========================================
# 4. PyTorch Neural Network Module
# ==========================================
class FullZZQuantumMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializing weights from 0 to 2*pi is best practice for quantum rotations
        self.conv1_rot = nn.Parameter(torch.rand(8, 3) * 2 * torch.pi)
        self.conv1_zz = nn.Parameter(torch.rand(28) * 2 * torch.pi)
        self.pool_w = nn.Parameter(torch.rand(4, 3) * 2 * torch.pi)
        self.conv2_rot = nn.Parameter(torch.rand(4, 3) * 2 * torch.pi)
        self.conv2_zz = nn.Parameter(torch.rand(6) * 2 * torch.pi)

    def forward(self, x):
        # OPTIMIZATION: PennyLane supports parameter broadcasting. 
        # By passing the entire batch 'x' directly, we eliminate the Python loop.
        # x shape: (batch_size, 256) -> quantum_probs shape: (batch_size, 16)
        quantum_probs = full_zz_qcnn(
            x, 
            self.conv1_rot, self.conv1_zz, 
            self.pool_w, 
            self.conv2_rot, self.conv2_zz
        )
        
        # OPTIMIZATION: Vectorized post-processing across the batch dimension
        class_probs = quantum_probs[:, :10]
        
        # Re-normalize using dim=1 to keep batch structure
        sum_probs = torch.sum(class_probs, dim=1, keepdim=True)
        normalized_probs = class_probs / (sum_probs + 1e-9)
        log_probs = torch.log(normalized_probs + 1e-9)
            
        return log_probs

# ==========================================
# 5. Training and Evaluation Pipeline
# ==========================================
def train_model(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader):
            # OPTIMIZATION: Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            # OPTIMIZATION: Move data to GPU
            images, labels = images.to(device), labels.to(device)
            
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
    # OPTIMIZATION: Increased batch size to 256. A 24GB GPU can easily handle 512+ for 8 qubits.
    BATCH_SIZE = 256  
    EPOCHS = 15
    LEARNING_RATE = 0.01

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device.type.upper()}")

    print("Preparing 16x16 MNIST Dataloaders...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    print("Initializing Full ZZ Quantum Model...")
    # OPTIMIZATION: Move model to device
    model = FullZZQuantumMNIST().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Total Trainable Quantum Parameters: {sum(p.numel() for p in model.parameters())}")

    print("Starting Training Loop...")
    train_model(model, train_loader, optimizer, criterion, EPOCHS, device)

    print("Starting Evaluation Loop...")
    evaluate_model(model, test_loader, device)
