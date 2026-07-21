import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ==========================================
# 1. High-Throughput Data Loading
# ==========================================
def prepare_mnist_dataloaders(batch_size: int):
    """
    Downloads the full MNIST dataset (60000 train / 10000 test), 
    resizes to 16x16, and flattens to 256-dimensional vectors.
    Optimized with pin_memory and multiprocessing for GPU transfer.
    """
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Architecture Functions
# ==========================================
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

def universal_two_qubit_gate(params, wires):
    """
    15-parameter decomposition of the universal SU(4) gate using {CNOT, RY, RZ}.
    Reflects the local connectivity filter for the convolutional layer.
    """
    qml.Rot(params[0], params[1], params[2], wires=wires[0])
    qml.Rot(params[3], params[4], params[5], wires=wires[1])
    
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[6], wires=wires[0])
    qml.RY(params[7], wires=wires[1])
    
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[1])
    
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.Rot(params[9], params[10], params[11], wires=wires[0])
    qml.Rot(params[12], params[13], params[14], wires=wires[1])

def local_quantum_conv_layer(params, wires):
    """
    Applies the universal two-qubit gate to neighboring qubits to mimic 
    a classical convolutional filter. The parameters are shared across all pairs.
    """
    n = len(wires)
    for i in range(n):
        universal_two_qubit_gate(params, [wires[i], wires[(i + 1) % n]])

def quantum_pooling_layer(params, source_wires, target_wires):
    """
    Applies classically controlled operations by invoking the principle 
    of deferred measurement. The outcome of the source determines the target rotation.
    """
    for src, tgt in zip(source_wires, target_wires):
        qml.CRot(params[0], params[1], params[2], wires=[src, tgt])

# ==========================================
# 3. Hybrid Quantum Node (QNode)
# ==========================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def hybrid_qcnn_circuit(inputs, conv_params, pool_params, fc_params):
    # Amplitude Encoding for 16x16 image (256 pixels -> 8 qubits)
    qml.AmplitudeEmbedding(features=inputs, wires=range(8), normalize=True)
    
    # --- Local Quantum Convolutional Layer ---
    local_quantum_conv_layer(conv_params, wires=range(8))
    
    # --- Quantum Pooling Layer ---
    quantum_pooling_layer(pool_params, source_wires=[1, 3, 5, 7], target_wires=[0, 2, 4, 6])
    
    # --- Quantum Fully Connected Layer ---
    qml.StronglyEntanglingLayers(fc_params, wires=[0, 2, 4, 6])
    
    # --- Hybrid Readout ---
    return [qml.expval(qml.PauliZ(w)) for w in [0, 2, 4, 6]]

# ==========================================
# 4. PyTorch Hybrid Model Module
# ==========================================
class HybridQCNNMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_params = nn.Parameter(torch.rand(15) * 2 * torch.pi)
        self.pool_params = nn.Parameter(torch.rand(3) * 2 * torch.pi)
        self.fc_params = nn.Parameter(torch.rand(1, 4, 3) * 2 * torch.pi)
        
        self.classical_dense = nn.Linear(in_features=4, out_features=10)

    def forward(self, x):
        # The QNode natively broadcasts across the batch dimension.
        quantum_expectations = torch.stack(
            hybrid_qcnn_circuit(x, self.conv_params, self.pool_params, self.fc_params)
        ).T 
        
        logits = self.classical_dense(quantum_expectations)
        return logits

# ==========================================
# 5. Training and Evaluation Pipeline
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    for epoch in range(epochs):
        
        # --- Training Phase ---
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track training accuracy
            _, predicted_classes = torch.max(outputs, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted_classes == labels).sum().item()
            
            # Print batch progress every 20 batches
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100.0
        
        # --- Validation Phase ---
        model.eval()
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                val_outputs = model(val_images)
                _, predicted_classes = torch.max(val_outputs, dim=1)
                
                total_val += val_labels.size(0)
                correct_val += (predicted_classes == val_labels).sum().item()
                
        val_accuracy = (correct_val / total_val) * 100.0
        
        print(f"--- Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% ---")

def evaluate_model(model, dataloader, device):
    """Final dedicated evaluation on the test set."""
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, dim=1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            
    accuracy = (correct_predictions / total_samples) * 100.0
    print("=====================================")
    print(f"Final Hybrid QCNN Test Accuracy: {accuracy:.2f}%")
    print("=====================================")
    return accuracy

# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 512 
    EPOCHS = 50
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device.type.upper()}")

    print("Preparing Full MNIST Dataloaders (16x16)...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    print("Initializing Hybrid QCNN Model...")
    model = HybridQCNNMNIST().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters (Quantum + Classical): {total_params}")

    print("\nStarting Training Loop...")
    train_model(model, train_loader, test_loader, optimizer, criterion, EPOCHS, device)

    print("\nStarting Final Evaluation Loop...")
    evaluate_model(model, test_loader, device)
