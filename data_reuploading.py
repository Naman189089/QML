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
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, test_loader

# ==========================================
# 2. Quantum Architecture
# ==========================================
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

def full_zz_block(rot_weights, zz_weights, wires):
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
    for i in range(num_qubits):
        qml.RY(inputs[:, i] * torch.pi, wires=i)

    for d in range(2):
        full_zz_block(l1_rot[d], l1_zz[d], wires=[0, 1, 2, 3])
            
    qml.CRot(pool_w[0, 0], pool_w[0, 1], pool_w[0, 2], wires=[1, 0])
    qml.CRot(pool_w[1, 0], pool_w[1, 1], pool_w[1, 2], wires=[3, 2])

    for d in range(2):
        full_zz_block(l2_rot[d], l2_zz[d], wires=[0, 2])
            
    return [qml.expval(qml.PauliZ(w)) for w in [0, 2]]

def extract_and_reshape_patches(images_batch):
    batch_size = images_batch.shape[0]
    patches = F.unfold(images_batch, kernel_size=2, stride=2) 
    patches_reshaped = patches.transpose(1, 2).reshape(batch_size * 64, 4)
    return patches_reshaped, batch_size


num_fc_qubits = 10  # 1 Qubit per MNIST class (0 to 9)
dev_fc = qml.device("default.qubit", wires=num_fc_qubits)

# ==========================================
# 2. Data Re-uploading Circuit
# ==========================================
@qml.qnode(dev_fc, interface="torch", diff_method="backprop")
def data_reuploading_circuit(inputs, var_weights, input_weights):
    """
    inputs: Tensor of shape (batch_size, 128)
    var_weights: Tensor of shape (num_layers, num_fc_qubits, 3) - The bias/shift
    input_weights: Tensor of shape (num_layers, num_fc_qubits, 3) - The scaling factor
    """
    num_layers = var_weights.shape[0]
    num_features = inputs.shape[1]
    
    for layer in range(num_layers):
        for i in range(num_fc_qubits):
            # Map specific features to this specific rotation gate.
            # Modulo operator ensures we wrap around if we run out of the 128 features.
            idx_x = (layer * 30 + i * 3 + 0) % num_features
            idx_y = (layer * 30 + i * 3 + 1) % num_features
            idx_z = (layer * 30 + i * 3 + 2) % num_features
            
            x_val = inputs[:, idx_x]
            y_val = inputs[:, idx_y]
            z_val = inputs[:, idx_z]
            
            # Weighted Data Re-uploading Step (alpha = theta + w * x)
            # PennyLane natively broadcasts the batch dimension for parameterized gates
            qml.RX(var_weights[layer, i, 0] + input_weights[layer, i, 0] * x_val, wires=i)
            qml.RY(var_weights[layer, i, 1] + input_weights[layer, i, 1] * y_val, wires=i)
            qml.RZ(var_weights[layer, i, 2] + input_weights[layer, i, 2] * z_val, wires=i)
            
        # Entanglement Layer: Ring Topology to correlate the classes
        for i in range(num_fc_qubits):
            qml.CNOT(wires=[i, (i + 1) % num_fc_qubits])
            
    # Measure expectation values on each qubit -> 10 output logits
    return [qml.expval(qml.PauliZ(i)) for i in range(num_fc_qubits)]

# ==========================================
# 3. PyTorch Module Wrapper
# ==========================================
class DataReuploadingFCLayer(nn.Module):
    def __init__(self, num_features=128, num_classes=10):
        super().__init__()
        
        # 10 qubits * 3 rotations per qubit = 30 features per layer
        # 128 / 30 = 4.26 -> We need 5 layers to upload all features at least once
        self.num_layers = 5 
        self.num_classes = num_classes
        
        # theta: Variational shift parameters (initialized uniformly)
        self.var_weights = nn.Parameter(torch.rand(self.num_layers, self.num_classes, 3) * 2 * torch.pi)
        
        # w: Input scaling weights (initialized to 1 so gradients flow immediately)
        self.input_weights = nn.Parameter(torch.ones(self.num_layers, self.num_classes, 3))

    def forward(self, x):
        # x shape: (batch_size, 128)
        # The QNode returns a list of 10 batched tensors. 
        # torch.stack packs them into (10, batch_size), and .T transposes to (batch_size, 10)
        q_out = torch.stack(data_reuploading_circuit(x, self.var_weights, self.input_weights)).T
        return q_out

class QuantumHeavyQuanvNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 4-Qubit Quanvolutional Filter Variables
        self.l1_rot = nn.Parameter(torch.rand(2, 4, 3) * 2 * torch.pi)
        self.l1_zz = nn.Parameter(torch.rand(2, 6) * 2 * torch.pi)
        self.pool_w = nn.Parameter(torch.rand(2, 3) * 2 * torch.pi)
        self.l2_rot = nn.Parameter(torch.rand(2, 2, 3) * 2 * torch.pi)
        self.l2_zz = nn.Parameter(torch.rand(2, 1) * 2 * torch.pi)
        
        # Replaced classical nn.Linear(128, 10) with Quantum Data Re-uploading
        self.quantum_fc = DataReuploadingFCLayer(num_features=128, num_classes=10)

    def forward(self, x):
        batched_patches, batch_size = extract_and_reshape_patches(x)
        
        quantum_out = torch.stack(
            deep_quantum_pooled_filter(batched_patches, self.l1_rot, self.l1_zz, self.pool_w, self.l2_rot, self.l2_zz)
        ).T 
        
        flattened_features = quantum_out.reshape(batch_size, 128)
        
        # Pass features through the Quantum FC layer
        logits = self.quantum_fc(flattened_features)
        return logits

# ==========================================
# 3. PGD Adversarial Attack Generator
# ==========================================
def pgd_attack(model, images, labels, epsilon, alpha, iters, device):
    """
    Generates adversarial examples using Projected Gradient Descent (PGD).
    Standard CE loss is used here to find the adversarial perturbation.
    """
    original_images = images.clone().detach()
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(original_images + eta, min=0, max=1)
            
        adv_images = adv_images.detach()

    return adv_images

# ==========================================
# 4. Training and Evaluation Pipelines
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, device, 
                is_adv_training=False, adv_eps=0.3, adv_alpha=0.075, adv_iters=10, lambda_param=0.4):
    
    # Initialize the KL Divergence criterion using 'batchmean' as recommended by PyTorch
    kl_criterion = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            if is_adv_training:
                # 1. Generate adversarial images
                model.eval()
                adv_images = pgd_attack(model, images, labels, adv_eps, adv_alpha, adv_iters, device)
                model.train()
                
                # 2. Forward pass for both Clean and Adversarial images
                logits_clean = model(images)
                logits_adv = model(adv_images)
                
                # 3. Calculate L_CE on clean sample
                loss_ce = criterion(logits_clean, labels)
                
                # 4. Calculate L_KL(softmax(clean), log_softmax(adv))
                # PyTorch KLDiv expects input in log-space, target in prob-space
                log_probs_adv = F.log_softmax(logits_adv, dim=1)
                probs_clean = F.softmax(logits_clean, dim=1)
                loss_kl = kl_criterion(log_probs_adv, probs_clean)
                
                # 5. Total Loss Combination
                loss = loss_ce + (lambda_param * loss_kl)
                
                # Track accuracy using the clean logits
                _, predicted_classes = torch.max(logits_clean, dim=1)
                
            else:
                # Standard Training
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted_classes = torch.max(outputs, dim=1)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_train += labels.size(0)
            correct_train += (predicted_classes == labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = (correct_train / total_train) * 100.0
        
        # Validation Phase
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
        
        print(f"--- Epoch {epoch+1} Summary | LR: {current_lr:.5f} | Train Acc: {train_accuracy:.2f}% | Clean Val Acc: {val_accuracy:.2f}% ---")


def evaluate_robustness(model, dataloader, device, attack_name="Clean", epsilon=0.0, alpha=0.0, iters=10):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    print(f"\nRunning {attack_name} Evaluation...")
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        if epsilon > 0.0:
            images = pgd_attack(model, images, labels, epsilon, alpha, iters, device)
            
        with torch.no_grad():
            outputs = model(images)
            _, predicted_classes = torch.max(outputs, dim=1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted_classes == labels).sum().item()
            
    accuracy = (correct_predictions / total_samples) * 100.0
    print(f">> {attack_name} Test Accuracy: {accuracy:.2f}%")
    return accuracy

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 512 
    EPOCHS = 50
    LEARNING_RATE = 0.005
    LAMBDA_PARAM = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device.type.upper()}")

    print("Preparing Datasets...")
    train_loader, test_loader = prepare_mnist_dataloaders(batch_size=BATCH_SIZE)

    # ---------------------------------------------------------
    # STAGE 1: STANDARD TRAINING
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("STAGE 1: STANDARD TRAINING")
    print("="*50)
    
    model_standard = QuantumHeavyQuanvNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_std = torch.optim.Adam(model_standard.parameters(), lr=LEARNING_RATE)
    scheduler_std = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_std, mode='min', factor=0.5, patience=5)

    print("\nTraining Standard Model...")
    train_model(model_standard, train_loader, test_loader, optimizer_std, scheduler_std, criterion, EPOCHS, device, 
                is_adv_training=False)

    evaluate_robustness(model_standard, test_loader, device, attack_name="Clean")
    evaluate_robustness(model_standard, test_loader, device, attack_name="PGD-10 (eps=0.3)", epsilon=0.3, alpha=0.075)

    # ---------------------------------------------------------
    # STAGE 2: KL-REGULARIZED ADVERSARIAL TRAINING
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(f"STAGE 2: KL-REGULARIZED ADVERSARIAL TRAINING (Lambda = {LAMBDA_PARAM})")
    print("="*50)
    
    model_robust = QuantumHeavyQuanvNN().to(device)
    optimizer_rob = torch.optim.Adam(model_robust.parameters(), lr=LEARNING_RATE)
    scheduler_rob = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_rob, mode='min', factor=0.5, patience=5)

    print("\nTraining Robust Model (Expect longer step times)...")
    train_model(model_robust, train_loader, test_loader, optimizer_rob, scheduler_rob, criterion, EPOCHS, device, 
                is_adv_training=True, adv_eps=0.3, adv_alpha=0.075, adv_iters=10, lambda_param=LAMBDA_PARAM)

    evaluate_robustness(model_robust, test_loader, device, attack_name="Clean")
    evaluate_robustness(model_robust, test_loader, device, attack_name="PGD-10 (eps=0.3)", epsilon=0.3, alpha=0.075)
    evaluate_robustness(model_robust, test_loader, device, attack_name="PGD-10 (eps=0.6)", epsilon=0.6, alpha=0.15)
    evaluate_robustness(model_robust, test_loader, device, attack_name="PGD-10 (eps=1.0)", epsilon=1.0, alpha=0.25)
