
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score


def load_data():
    """
    Load the shared, preprocessed training, validation, and test datasets.
    """

    # Find the project root and the folder containing processed data.
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    # Load the prepared datasets created by the shared preprocessing notebook.
    train_df = pd.read_csv(
        data_dir / "humanitarian_train.tsv",
        sep="\t",
        dtype={"id": "string"},
    )
    dev_df = pd.read_csv(
        data_dir / "humanitarian_dev.tsv",
        sep="\t",
        dtype={"id": "string"},
    )
    test_df = pd.read_csv(
        data_dir / "humanitarian_test.tsv",
        sep="\t",
        dtype={"id": "string"},
    )

    # Check that the expected model input and target columns exist.
    required_columns = {"text_clean", "class_label"}
    for name, df in [
        ("training", train_df),
        ("validation", dev_df),
        ("test", test_df),
    ]:
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {name} data: {missing}"
            )

    return train_df, dev_df, test_df


def prepare_labels(train_df, dev_df, test_df):
    """
    Convert humanitarian category names into consistent numerical labels.
    """

    # Create the mapping using training labels only.
    label_names = sorted(train_df["class_label"].unique())
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    # Ensure validation and test labels are known to the model.
    for name, df in [("validation", dev_df), ("test", test_df)]:
        unknown_labels = set(df["class_label"].unique()) - set(label2id)
        if unknown_labels:
            raise ValueError(
                f"Unknown categories in {name} data: {unknown_labels}"
            )

    # Copy the data so the original DataFrames remain unchanged.
    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    # Add numerical labels while preserving the original category names.
    for df in (train_df, dev_df, test_df):
        df["label"] = df["class_label"].map(label2id)

    return train_df, dev_df, test_df, label2id, id2label


class CrisisDataset(Dataset):
    """
    Store tokenized social media posts and their labels for PyTorch.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        # Return the number of posts in the dataset.
        return len(self.labels)

    def __getitem__(self, index):
        # Convert the tokenized inputs for one post into tensors.
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }

        # BERT expects classification labels to be integer tensors.
        item["labels"] = torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )

        return item


def tokenize_data(train_df, dev_df, test_df, tokenizer):
    """
    Tokenize the shared text_clean field and create PyTorch datasets.
    """

    max_length = 128
    datasets = []

    for df in (train_df, dev_df, test_df):
        # Use the cleaned text specified by the shared project workflow.
        encodings = tokenizer(
            df["text_clean"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        dataset = CrisisDataset(
            encodings,
            df["label"].tolist(),
        )
        datasets.append(dataset)

    return tuple(datasets)



def train_model(train_dataset, dev_dataset, label2id, id2label):
    """
    Fine-tune BERT on the full training dataset.
    Select the best model using validation macro F1 and stop
    early if validation performance stops improving.
    """
    import copy
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import f1_score
    import torch.nn.functional as F

    # Training configuration
    model_name = "bert-base-uncased"
    batch_size = 8
    epochs = 3
    learning_rate = 2e-5
    patience = 2

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Training device: {device}", flush=True)
    print(f"Training examples: {len(train_dataset)}", flush=True)
    print(f"Validation examples: {len(dev_dataset)}", flush=True)

    # Load the full datasets. Do not use Subset here.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Load pretrained BERT with a classification head for 11 categories.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Give less frequent classes more influence during training.
    # Square-root weighting avoids excessively large weights.
    train_labels = torch.as_tensor(
        train_dataset.labels, dtype=torch.long
    )
    class_counts = torch.bincount(
        train_labels, minlength=len(label2id)
    ).float()

    if (class_counts == 0).any():
        raise ValueError(
            "At least one category is missing from the training data."
        )

    class_weights = (
        class_counts.sum() / (len(label2id) * class_counts)
    ).sqrt()
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            labels = batch.pop("labels")

            optimizer.zero_grad(set_to_none=True)

            outputs = model(**batch)

            # Weighted cross-entropy helps account for
            # differences in category frequencies.
            loss = F.cross_entropy(
                outputs.logits,
                labels,
                weight=class_weights,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"{batch_idx + 1}/{len(train_loader)} "
                    f"training batches completed",
                    flush=True,
                )

        avg_train_loss = total_loss / len(train_loader)

        # -------------------------
        # Full validation
        # -------------------------
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dev_loader):
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                labels = batch.pop("labels")
                outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=1)

                all_predictions.extend(
                    predictions.cpu().tolist()
                )
                all_labels.extend(labels.cpu().tolist())

                if (batch_idx + 1) % 100 == 0:
                    print(
                        f"Validation: {batch_idx + 1}/"
                        f"{len(dev_loader)} batches completed",
                        flush=True,
                    )

        val_macro_f1 = f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_macro_f1": val_macro_f1,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} complete | "
            f"Training loss: {avg_train_loss:.4f} | "
            f"Validation macro F1: {val_macro_f1:.4f}",
            flush=True,
        )

        # Save the best-performing model.
        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            best_state = copy.deepcopy({
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            })
            epochs_without_improvement = 0
            print("New best model saved in memory.", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for "
                f"{epochs_without_improvement} epoch(s).",
                flush=True,
            )

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.", flush=True)
            break

    # Restore the weights from the best validation epoch.
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_state)
    model.to(device)

    print(f"Best validation macro F1: {best_f1:.4f}", flush=True)

    return model, history




def predict(model, dataset):
    """
    Generate class predictions using a trained BERT model.
    Returns a list of predicted numeric class IDs.
    """
    import torch
    from torch.utils.data import DataLoader

    device = next(model.parameters()).device
    model.eval()

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # Labels are not needed to generate predictions.
            batch.pop("labels", None)

            outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=1)

            predictions.extend(batch_predictions.cpu().tolist())

    return predictions



def save_predictions(predictions, output_path):
    """
    Save predicted class IDs to a CSV file for evaluation
    and comparison with the baseline.
    """
    import pandas as pd
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame({
        "prediction": predictions
    })

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved {len(predictions)} predictions to {output_path}")

