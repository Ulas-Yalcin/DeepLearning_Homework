from pathlib import Path

# Yolları çalıştırma dizinine değil, bu projeye göre kuruyorum.
PROJECT_DIR = Path(__file__).resolve().parent.parent
SEED = 42
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "results"
HANDWRITING_DIR = PROJECT_DIR / "handwriting"

# Deneylerde gizli katman genişliği değişiyor; diğer ayarlar sabit.
MODEL_CONFIGS = {
    "mlp_128_64": [128, 64],
    "mlp_256_128": [256, 128],
    "mlp_512_256": [512, 256],
}
