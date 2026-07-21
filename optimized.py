import torch
import torch.nn as nn
import pennylane as qml

# ==========================================
# 0. Device / dtype setup
# ==========================================
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)  # complex64 internally instead of complex128

n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)


# ==========================================
# 1. Circuit building blocks (logic unchanged —
#    Rot / IsingZZ / CRot / AmplitudeEmbedding all already
#    support an optional leading batch dimension)
# ==========================================
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
# 2. Batched QNode
#    diff_method="backprop" is REQUIRED for execution to follow
#    the device (CPU/CUDA) of the input tensors.
#    `inputs` is now (batch, 256) — the WHOLE batch runs as one
#    broadcasted circuit call, no Python loop.
# ==========================================
@qml.qnode(dev, interface="torch", diff_method="backprop")
def full_zz_qcnn(inputs, conv1_rot, conv1_zz, pool_w, conv2_rot, conv2_zz):
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True)

    full_zz_entanglement_layer(conv1_rot, conv1_zz, wires=range(8))
    quantum_pooling_layer(pool_w, source_wires=[1, 3, 5, 7], target_wires=[0, 2, 4, 6])
    full_zz_entanglement_layer(conv2_rot, conv2_zz, wires=[0, 2, 4, 6])

    return qml.probs(wires=[0, 2, 4, 6])  # shape (batch, 16)


# ==========================================
# 3. PyTorch Module
# ==========================================
class FullZZQuantumMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_rot = nn.Parameter(torch.rand(8, 3))
        self.conv1_zz = nn.Parameter(torch.rand(28))
        self.pool_w = nn.Parameter(torch.rand(4, 3))
        self.conv2_rot = nn.Parameter(torch.rand(4, 3))
        self.conv2_zz = nn.Parameter(torch.rand(6))

    def forward(self, x):
        """
        x: (batch, 256) real-valued tensor, already on the same device
        as the model. No per-sample Python loop.
        """
        quantum_probs = full_zz_qcnn(
            x,
            self.conv1_rot, self.conv1_zz,
            self.pool_w,
            self.conv2_rot, self.conv2_zz
        )  # (batch, 16)

        class_probs = quantum_probs[:, :10]
        sum_probs = class_probs.sum(dim=1, keepdim=True)
        normalized_probs = class_probs / (sum_probs + 1e-9)
        log_probs = torch.log(normalized_probs + 1e-9)
        return log_probs


# ==========================================
# 4. Usage / training loop skeleton
# ==========================================
if __name__ == "__main__":
    model = FullZZQuantumMNIST().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.NLLLoss()

    # 8->4 qubits is memory-trivial; 24GB VRAM is not the constraint here.
    # Kernel-launch/tracing overhead dominates instead, so push batch size
    # up and benchmark CPU vs GPU directly rather than assuming GPU wins.
    batch_size = 1024
    x = torch.rand(batch_size, 256, device=torch_device)          # replace with real data
    y = torch.randint(0, 10, (batch_size,), device=torch_device)  # replace with real labels

    optimizer.zero_grad()
    log_probs = model(x)          # (batch, 10)
    loss = loss_fn(log_probs, y)
    loss.backward()
    optimizer.step()

    print("output shape:", log_probs.shape)
    print("loss:", loss.item())
