import time

import torch
from torch import nn

from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from .model import MLP


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_one_model(name, hidden_sizes, loaders, device):
    model = MLP(hidden_sizes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, loaders["train_eval"], loss_fn, device)
        val_loss, val_acc = evaluate(model, loaders["validation"], loss_fn, device)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_accuracy"].append(val_acc)
        print(
            f"{name} | epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"train acc {train_acc:.4f} | val acc {val_acc:.4f}"
        )

    elapsed = time.perf_counter() - start
    summary = {
        "model": name,
        "hidden_layers": hidden_sizes,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["validation_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_validation_accuracy": history["validation_accuracy"][-1],
        "training_seconds": round(elapsed, 2),
    }
    return model, history, summary
