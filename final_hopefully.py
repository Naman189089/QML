import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. High-Throughput Data Loading
# ==========================================
def prepare_mnist_dataloaders(batch_size: int, max_train_samples: int = 60000, max_test_samples: int = 10000):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
        # No flattening here; 2D structure is required for sliding patches.
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Device & Deeper Quanv Filter
# ==========================================
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

def full_zz_block(rot_weights, zz_weights, wires):
    """
    Explicit helper function for a single Full ZZ Entanglement layer.
    """
    n = len(wires)
    for i in range(n):
        qml.Rot(rot_weights[i, 0], rot_weights[i, 1], rot_weights[i, 2], wires=wires[i])
        
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ(zz_weights[idx], wires=[wires[i], wires[j]])
            idx += 1

@qml.qnode(dev, interface="torch", diff_method="backprop")
def deep_quantum_pooled_filter(inputs, l1_rot, l1_zz, pool_w, l2_rot, l2_zz):
    """
    Deep 4-Qubit Quanvolutional Filter with Pooling.
    Iterates the entanglement blocks (Depth=2) to increase quantum expressivity.
    """
    # 1. Data Encoding
    for i in range(num_qubits):
        qml.RY(inputs[:, i] * torch.pi, wires=i)

    # 2. Pre-Pooling Layer (Depth 2: repeated Full ZZ blocks)
    for d in range(2):
        full_zz_block(l1_rot[d], l1_zz[d], wires=[0, 1, 2, 3])
            
    # 3. Quantum Pooling (4 Qubits -> 2 Qubits)
    qml.CRot(pool_w[0, 0], pool_w[0, 1], pool_w[0, 2], wires=[1, 0])
    qml.CRot(pool_w[1, 0], pool_w[1, 1], pool_w[1, 2], wires=[3, 2])

    # 4. Post-Pooling Layer (Depth 2: repeated Full ZZ blocks on Wires 0 & 2)
    for d in range(2):
        full_zz_block(l2_rot[d], l2_zz[d], wires=[0, 2])
            
    # 5. Measurement
    return [qml.expval(qml.PauliZ(w)) for w in [0, 2]]

# ==========================================
# 3. Patch Extraction Helper
# ==========================================
def extract_and_reshape_patches(images_batch):
    batch_size = images_batch.shape[0]
    patches = F.unfold(images_batch, kernel_size=2, stride=2) 
    patches_reshaped = patches.transpose(1, 2).reshape(batch_size * 64, 4)
    return patches_reshaped, batch_size

# ==========================================
# 4. PyTorch Hybrid Module (Minimal Classical)
# ==========================================
class QuantumHeavyQuanvNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # QUANTUM PARAMETERS (Increased capacity due to depth=2)
        # l1: Depth 2, 4 wires, 3 rot params | Depth 2, 6 ZZ params
        self.l1_rot = nn.Parameter(torch.rand(2, 4, 3) * 2 * torch.pi)
        self.l1_zz = nn.Parameter(torch.rand(2, 6) * 2 * torch.pi)
        
        # pooling: 2 target wires, 3 rot params
        self.pool_w = nn.Parameter(torch.rand(2, 3) * 2 * torch.pi)
        
        # l2: Depth 2, 2 wires, 3 rot params | Depth 2, 1 ZZ param
        self.l2_rot = nn.Parameter(torch.rand(2, 2, 3) * 2 * torch.pi)
        self.l2_zz = nn.Parameter(torch.rand(2, 1) * 2 * torch.pi)
        
        # CLASSICAL PARAMETERS (Strictly constrained to final classification)
        # 64 patches * 2 output expectation values = 128 features mapped to 10 classes
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 1. Extract Patches
        batched_patches, batch_size = extract_and_reshape_patches(x)
        
        # 2. Deep Quantum Filter
        quantum_out = torch.stack(
            deep_quantum_pooled_filter(batched_patches, self.l1_rot, self.l1_zz, self.pool_w, self.l2_rot, self.l2_zz)
        ).T 
        
        # 3. Reshape and Flatten
        flattened_features = quantum_out.reshape(batch_size, 128)
        
        # 4. Direct Classical Classification (No hidden layers)
        logits = self.fc(flattened_features)
        
        return logits

# ==========================================
# 5. Training and Evaluation Pipeline
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device):
    for epoch in range(epochs):
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
            _, predicted_classes = torch.max(outputs, dim=1)
            total_train += labels.size(0)
            correct_train += (predicted_classes == labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100.0
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                
                val_outputs = model(val_images)
                batch_loss = criterion(val_outputs, val_labels)
                val_loss += batch_loss.item()
                
                _, predicted_classes = torch.max(val_outputs, dim=1)
                total_val += val_labels.size(0)
                correct_val += (predicted_classes == val_labels).sum().item()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_val / total_val) * 100.0
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"--- Epoch {epoch+1} Summary | LR: {current_lr:.5f} | Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}% ---")

def evaluate_model(model, dataloader, device):
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
    print(f"Final Quantum-Heavy QuanvNN Test Accuracy: {accuracy:.2f}%")
    print("=====================================")
    return accuracy

# ==========================================
# 6. Main Execution Block
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 512 
    EPOCHS = 50
    LEARNING_RATE = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device.type.upper()}")

    print("Preparing Datasets...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    print("Initializing Quantum-Heavy Quanvolutional Network...")
    model = QuantumHeavyQuanvNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    total_q_params = sum(p.numel() for name, p in model.named_parameters() if 'fc' not in name)
    total_c_params = sum(p.numel() for name, p in model.named_parameters() if 'fc' in name)
    print(f"Total Trainable Quantum Parameters: {total_q_params}")
    print(f"Total Trainable Classical Parameters: {total_c_params}")

    print("\nStarting Training Loop...")
    train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, EPOCHS, device)

    print("\nStarting Final Evaluation Loop...")
    evaluate_model(model, test_loader, device)
