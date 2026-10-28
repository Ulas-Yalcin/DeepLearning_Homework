import csv
import json
from pathlib import Path

import torch

from mnist_mlp.config import EPOCHS, LEARNING_RATE, MODEL_CONFIGS, OUTPUT_DIR, SEED
from mnist_mlp.data import make_data_loaders, set_seed
from mnist_mlp.handwriting import predict_folder
from mnist_mlp.model import MLP
from mnist_mlp.training import evaluate, train_one_model
from mnist_mlp.visualization import save_training_curves


def main():
    set_seed()
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    loaders, data_sizes = make_data_loaders()

    histories = {}
    summaries = []
    best_validation_accuracy = -1.0
    best_checkpoint_path = OUTPUT_DIR / "best_mlp.pt"

    for name, hidden_sizes in MODEL_CONFIGS.items():
        print(f"\nEğitim başlıyor: {name} — gizli katmanlar {hidden_sizes}")
        model, history, summary = train_one_model(name, hidden_sizes, loaders, device)
        histories[name] = history
        summaries.append(summary)
        if summary["final_validation_accuracy"] > best_validation_accuracy:
            best_validation_accuracy = summary["final_validation_accuracy"]
            torch.save({"model": model.state_dict(), "hidden_sizes": hidden_sizes, "name": name}, best_checkpoint_path)

    # Test kümesi yalnızca validation ile model seçildikten sonra kullanılır.
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    best_model = MLP(checkpoint["hidden_sizes"]).to(device)
    best_model.load_state_dict(checkpoint["model"])
    test_loss, test_accuracy = evaluate(
        best_model, loaders["test"], torch.nn.CrossEntropyLoss(), device
    )
    selected = next(row for row in summaries if row["model"] == checkpoint["name"])
    selected["test_loss"] = test_loss
    selected["test_accuracy"] = test_accuracy

    fields = [
        "model", "hidden_layers", "epochs", "batch_size", "learning_rate", "optimizer",
        "final_train_loss", "final_validation_loss", "final_train_accuracy",
        "final_validation_accuracy", "training_seconds", "test_loss", "test_accuracy",
    ]
    with (OUTPUT_DIR / "model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    metrics = {
        "selected_model": checkpoint["name"],
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "data_sizes": data_sizes,
        "experiments": summaries,
    }
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    save_training_curves(
        histories[checkpoint["name"]], checkpoint["name"], OUTPUT_DIR / "training_curves.png"
    )
    print("\nDeney özeti:")
    for row in summaries:
        print(f"{row['model']}: validation accuracy={row['final_validation_accuracy']:.4f}, "
              f"eğitim süresi={row['training_seconds']} sn")
    print(f"Seçilen modelin test accuracy değeri: {test_accuracy:.4f}")
    print(f"Sonuç dosyaları: {Path(OUTPUT_DIR).resolve()}")
    predict_folder(best_model, device)


if __name__ == "__main__":
    main()
