import matplotlib.pyplot as plt


def save_training_curves(history, model_name, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Eğitim")
    axes[0].plot(epochs, history["validation_loss"], label="Doğrulama")
    axes[0].set(title="Kayıp (loss)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Eğitim")
    axes[1].plot(epochs, history["validation_accuracy"], label="Doğrulama")
    axes[1].set(title="Doğruluk (accuracy)", xlabel="Epoch", ylabel="Doğruluk")
    axes[1].legend()
    fig.suptitle(f"Seçilen model: {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
