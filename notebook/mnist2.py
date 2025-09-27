from pathlib import Path
from typing import List

import numpy as np
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from bitwise2 import fc, BitTensor
from bitwise2.model import Model
from bitwise2.util import map_bit_counts


def identity_matrix(m: int, n: int) -> torch.Tensor:
    """Creates an identity-like matrix for a given hight and width."""

    # Number of int32 elements needed to store n bits
    num_elts = (n + 31) // 32

    # Initialize tensor with zeros, using torch.int32
    tensor = torch.zeros((m, num_elts), dtype=torch.int32)

    # Compute row indices and bit positions, keeping indices as int64 for indexing
    row_indices = torch.arange(m, dtype=torch.int64)  # Shape: (m,)
    k = row_indices % n  # Bit position in logical range, Shape: (m,)
    elt_idx = k // 32  # Index of the int32 element, Shape: (m,)
    bit_pos = 31 - (k % 32)  # Bit position within int32, Shape: (m,)

    # Convert bit_pos to int32 for the shift operation
    bit_pos_int32 = bit_pos.to(torch.int32)

    # Compute bit values directly in int32, avoiding int64
    values = torch.ones(m, dtype=torch.int32) << bit_pos_int32

    # Assign values to the tensor using advanced indexing
    tensor[row_indices, elt_idx] = values

    return tensor


def create_untrained_model(layer_widths: List[int], device: str = "cpu") -> Model:
    """Create a random untrained model."""

    layers: List[fc.FullyConnectedLayer] = []

    for ins, outs in list(zip(layer_widths, layer_widths[1:])):
        data = identity_matrix(outs, ins)
        weights = BitTensor(ins, data.to(device=device))
        data = torch.randint(-(2**31), 2**31, ((outs + 31) // 32,), dtype=torch.int32)
        bias = BitTensor(outs, data.to(device=device))
        layer = fc.FullyConnectedLayer(weights, bias)
        layers.append(layer)

    return Model(layers)


def read_mnist_images(filename: Path, device: str = "cpu") -> torch.Tensor:
    with open(filename, "rb") as f:
        _, num_images, rows, cols = np.frombuffer(f.read(16), dtype=">u4")  # Big-endian
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return torch.tensor(images, device=device)


def read_mnist_labels(filename: Path, device: str = "str") -> torch.Tensor:
    with open(filename, "rb") as f:
        _, _ = np.frombuffer(f.read(8), dtype=">u4")  # Big-endian
        labels = np.frombuffer(f.read(), dtype=np.uint8)  # Labels are 1-byte each
    return torch.tensor(labels, device=device)


def create_dataset(
    images: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> TensorDataset:
    assert len(images) == len(labels)
    size = len(images) // batch_size * batch_size
    return TensorDataset(images[:size], labels[:size])


def convert_images_to_inputs(images: torch.Tensor) -> BitTensor:
    data = images.view(len(images), 28 * 28).view(torch.int32)
    return BitTensor(data.shape[-1] * 32, data)


def convert_labels_to_outputs(labels: torch.Tensor) -> BitTensor:
    data = (
        F.one_hot(labels.to(dtype=torch.long), num_classes=10).to(dtype=torch.int32)
        * -1
    )
    return BitTensor(10 * 32, data)


def get_class_probabilities(outputs: BitTensor) -> torch.Tensor:
    bit_counts = map_bit_counts(outputs.data)
    return F.softmax(bit_counts.to(dtype=torch.float32), dim=1)


def compute_accuracy(model: Model, dataset: TensorDataset, batch_size: int) -> float:
    correct: int = 0
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for images, labels in dataloader:
        inputs = convert_images_to_inputs(images)
        outputs = model.eval(inputs)
        predicted = get_class_probabilities(outputs).argmax(dim=-1)
        correct += (predicted == labels).sum().item()
    return correct / len(dataset)


mnist, device = Path("data/mnist"), "cuda"
train_images = read_mnist_images(mnist / "train-images.idx3-ubyte", device=device)
train_labels = read_mnist_labels(mnist / "train-labels.idx1-ubyte", device=device)
test_images = read_mnist_images(mnist / "t10k-images.idx3-ubyte", device=device)
test_labels = read_mnist_labels(mnist / "t10k-labels.idx1-ubyte", device=device)

model = create_untrained_model([28 * 28 * 8, 4096, 4096, 4096, 10 * 32], device=device)

batch_size = 1024
for epoch in range(100):
    train_dataset = create_dataset(train_images, train_labels, batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for i, (images, labels) in enumerate(train_dataloader):
        inputs = convert_images_to_inputs(images)
        expected_outputs = convert_labels_to_outputs(labels)
        outputs, updated_layers = model.update(inputs, expected_outputs)
        predicted = get_class_probabilities(outputs).argmax(dim=-1)
        correct = (predicted == labels).sum().item()
        progress = ((i + 1) / len(train_dataloader)) * 100
        print(
            f"{epoch}-{progress:.2f}% updated_layers={updated_layers} accuracy={correct/batch_size:.2f}"
        )

    test_dataset = create_dataset(test_images, test_labels, batch_size)
    accuracy = compute_accuracy(model, test_dataset, batch_size)
    print(f"epoch {epoch} accuracy {accuracy:.2f}")
