import csv

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from .config import HANDWRITING_DIR, OUTPUT_DIR


def prepare_image(path):
    """Görseli siyah zemin üzerinde beyaz rakamlı 28x28 MNIST biçimine yaklaştırır."""
    image = Image.open(path).convert("L")
    image.thumbnail((500, 500))
    pixels = np.asarray(image)
    corners = [pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]]
    if np.mean(corners) > 127:  # Açık kağıt zeminini koyu zemine çevir.
        image = ImageOps.invert(image)
    image = image.point(lambda value: 255 if value > 45 else 0)
    bbox = image.getbbox()
    if bbox is None:
        return Image.new("L", (28, 28), 0)
    digit = image.crop(bbox)
    digit.thumbnail((20, 20))
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return canvas


def predict_folder(model, device):
    image_paths = sorted(
        path for path in HANDWRITING_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ) if HANDWRITING_DIR.exists() else []
    if not image_paths:
        print("El yazısı görseli yok; handwriting klasörüne kendi rakam görsellerinizi ekleyebilirsiniz.")
        return

    rows = []
    model.eval()
    with torch.no_grad():
        for path in image_paths:
            tensor = transforms.ToTensor()(prepare_image(path)).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            prediction = int(probabilities.argmax().item())
            confidence = float(probabilities[prediction].item())
            rows.append([path.name, prediction, round(confidence, 4)])
            print(f"{path.name}: tahmin={prediction}, güven={confidence:.1%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "handwriting_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dosya", "tahmin", "guven"])
        writer.writerows(rows)
