import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ==========================================
# 1. CPU-Friendly Data Loading
# ==========================================
def prepare_mnist_dataloaders(batch_size: int, max_train_samples: int = 60000, max_test_samples: int = 10000):
    transform = transforms.Compose([
        transforms.Resize((16, 16)),
        transforms.ToTensor()
        # Preserving 2D spatial dimensions for the sliding filter
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Removed pin_memory and num_workers to optimize for standard CPU execution
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Devices & Explicit Blocks
# ==========================================
dev_filter = qml.device("default.qubit", wires=4)
dev_head = qml.device("default.qubit", wires=7)

def full_zz_block(rot_weights, zz_weights, wires):
    """Explicit helper for the Full ZZ Entanglement block."""
    n = len(wires)
    for i in range(n):
        qml.Rot(rot_weights[i, 0], rot_weights[i, 1], rot_weights[i, 2], wires=wires[i])
        
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            qml.IsingZZ(zz_weights[idx], wires=[wires[i], wires[j]])
            idx += 1

# ==========================================
# 3. The Two Quantum Nodes
# ==========================================
@qml.qnode(dev_filter, interface="torch", diff_method="backprop")
def deep_quantum_pooled_filter(inputs, l1_rot, l1_zz, pool_w, l2_rot, l2_zz):
    """QNode 1: The Sliding 4-Qubit Quanvolutional Feature Extractor"""
    # Angle Encoding for the 4-pixel patch
    for i in range(4):
        qml.RY(inputs[:, i] * torch.pi, wires=i)

    # Depth-2 Pre-Pooling
    for d in range(2):
        full_zz_block(l1_rot[d], l1_zz[d], wires=[0, 1, 2, 3])
            
    # Quantum Pooling
    qml.CRot(pool_w[0, 0], pool_w[0, 1], pool_w[0, 2], wires=[1, 0])
    qml.CRot(pool_w[1, 0], pool_w[1, 1], pool_w[1, 2], wires=[3, 2])

    # Depth-2 Post-Pooling
    for d in range(2):
        full_zz_block(l2_rot[d], l2_zz[d], wires=[0, 2])
            
    return [qml.expval(qml.PauliZ(w)) for w in [0, 2]]

@qml.qnode(dev_head, interface="torch", diff_method="backprop")
def quantum_classification_head(features, weights):
    """QNode 2: The 7-Qubit Amplitude-Encoded Classifier"""
    # Pack the 128 features into the 2^7 amplitudes of 7 qubits
    qml.AmplitudeEmbedding(features=features, wires=range(7), normalize=True)
    
    # Deeply entangle the 7 qubits to mix the 128 features globally
    qml.StronglyEntanglingLayers(weights, wires=range(7))
    
    # Return probabilities of the 128 basis states
    return qml.probs(wires=range(7))

# ==========================================
# 4. Patch Extraction Helper
# ==========================================
def extract_and_reshape_patches(images_batch):
    batch_size = images_batch.shape[0]
    patches = F.unfold(images_batch, kernel_size=2, stride=2) 
    patches_reshaped = patches.transpose(1, 2).reshape(batch_size * 64, 4)
    return patches_reshaped, batch_size

# ==========================================
# 5. Pure Q2Q Neural Network Module
# ==========================================
class PureQ2QMNIST(nn.Module):
    def __init__(self, head_layers=2):
        super().__init__()
        
        # --- QUANTUM FILTER PARAMETERS ---
        self.l1_rot = nn.Parameter(torch.rand(2, 4, 3) * 2 * torch.pi)
        self.l1_zz = nn.Parameter(torch.rand(2, 6) * 2 * torch.pi)
        self.pool_w = nn.Parameter(torch.rand(2, 3) * 2 * torch.pi)
        self.l2_rot = nn.Parameter(torch.rand(2, 2, 3) * 2 * torch.pi)
        self.l2_zz = nn.Parameter(torch.rand(2, 1) * 2 * torch.pi)
        
        # --- QUANTUM HEAD PARAMETERS ---
        # StronglyEntanglingLayers requires shape (layers, qubits, 3)
        self.head_weights = nn.Parameter(torch.rand(head_layers, 7, 3) * 2 * torch.pi)
        
        # Note: Zero classical nn.Linear parameters in this architecture!

    def forward(self, x):
        # 1. Classical routing: Extract patches
        batched_patches, batch_size = extract_and_reshape_patches(x)
        
        # 2. Sub-circuit 1: Feature Extraction
        quantum_out = torch.stack(
            deep_quantum_pooled_filter(batched_patches, self.l1_rot, self.l1_zz, self.pool_w, self.l2_rot, self.l2_zz)
        ).T 
        
        # 3. Classical routing: Reshape to (Batch, 128) and L2-Normalize
        flattened_features = quantum_out.reshape(batch_size, 128)
        features_normalized = F.normalize(flattened_features, p=2, dim=1)
        
        # 4. Sub-circuit 2: Amplitude classification
        head_probs = quantum_classification_head(features_normalized, self.head_weights)
        
        # 5. Pure Quantum Readout Slicing (digits 0-9)
        class_probs = head_probs[:, :10]
        
        # Re-normalize over the 10 target classes
        sum_probs = torch.sum(class_probs, dim=1, keepdim=True)
        normalized_probs = class_probs / (sum_probs + 1e-9)
        
        # Compute log-probabilities for NLLLoss
        log_probs = torch.log(normalized_probs + 1e-9)
        
        return log_probs

# ==========================================
# 6. CPU Training and Evaluation Pipeline
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs):
    # Hardcoded to CPU execution
    device = torch.device("cpu")
    model.to(device)
    
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
            
            # if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            #     print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
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

def evaluate_model(model, dataloader):
    device = torch.device("cpu")
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
    print(f"Final Pure Q2Q Test Accuracy: {accuracy:.2f}%")
    print("=====================================")
    return accuracy

# ==========================================
# 7. Main Execution Block
# ==========================================
if __name__ == "__main__":
    # DRASTICALLY REDUCED for CPU survival. 
    BATCH_SIZE = 512
    EPOCHS = 100
    LEARNING_RATE = 0.005

    print("EXECUTING ON: CPU (Expect long iteration times due to nested quantum circuits)")

    print("Preparing Datasets...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    print("Initializing Pure Quantum-to-Quantum Network...")
    # head_layers dictates the depth of the StronglyEntanglingLayers
    model = PureQ2QMNIST(head_layers=2)
    
    # Reverted to NLLLoss because the quantum head outputs normalized log-probabilities
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    total_q_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Quantum Parameters: {total_q_params}")
    print(f"Total Trainable Classical Parameters: 0")

    print("\nStarting Training Loop...")
    train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, EPOCHS)

    print("\nStarting Final Evaluation Loop...")
    evaluate_model(model, test_loader)
