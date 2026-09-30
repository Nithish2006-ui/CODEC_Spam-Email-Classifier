"""
Train Naive Bayes and SVM spam classifiers, compare them, and save the best.
Run: python train.py
"""

import os

import matplotlib
matplotlib.use("Agg")  # save figures to file, no display window needed
import matplotlib.pyplot as plt
import pandas as pd

from spam_classifier import BASE_DIR, load_data, save_model, train_and_compare


def plot_confusion_matrices(results: dict, path: str) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(9, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = res["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["ham", "spam"])
        ax.set_yticks([0, 1], ["ham", "spam"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    df = load_data()
    n_spam = int(df["target"].sum())
    print(f"Dataset: {len(df)} messages after cleaning "
          f"({n_spam} spam / {len(df) - n_spam} ham)\n")

    results, best, pipelines = train_and_compare(df)

    for name, res in results.items():
        print("=" * 55)
        print(f"{name.upper()}")
        print("=" * 55)
        print(res["report"])
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(res["confusion_matrix"], "\n")

    summary = pd.DataFrame(
        {n: {k: round(r[k], 4) for k in ("accuracy", "precision", "recall", "f1")}
         for n, r in results.items()}
    ).T
    print("=" * 55)
    print("MODEL COMPARISON (spam class)")
    print("=" * 55)
    print(summary, "\n")

    save_model(pipelines[best])
    print(f"Best model by F1: {best} -> saved to models/spam_model.joblib")

    out_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "confusion_matrices.png")
    plot_confusion_matrices(results, plot_path)
    print(f"Confusion matrix plot saved to outputs/confusion_matrices.png")


if __name__ == "__main__":
    main()
