
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_data():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "event_aware_en"

    # Load the training dataset.
    train_df = pd.read_csv(
        data_dir
        / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_train.tsv",
        sep="\t",
    )

    # Load the validation dataset.
    dev_df = pd.read_csv(
        data_dir
        / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_dev.tsv",
        sep="\t",
    )

    # Load the test dataset.
    test_df = pd.read_csv(
        data_dir
        / "crisis_consolidated_humanitarian_filtered_lang_en_w_event_info_test.tsv",
        sep="\t",
    )

    # Return all three DataFrames so other functions can use them.
    return train_df, dev_df, test_df


def prepare_labels(train_df, dev_df, test_df):
    """
    Convert humanitarian category names into numerical labels.

    BERT needs a number for each category rather than a text label.
    We create the mapping from the training data and reuse it for
    validation and test data to keep the labels consistent.
    """

    # Get the unique category names from the training data and sort them.
    label_names = sorted(train_df["class_label"].unique())

    # Map each category name to a number, starting at 0.
    label2id = {label: i for i, label in enumerate(label_names)}

    # Create the reverse mapping to convert numbers back to category names.
    id2label = {i: label for label, i in label2id.items()}

    # Check that validation and test data don't contain categories that were absent from the training data.
    for name, df in [("validation", dev_df), ("test", test_df)]:
        unknown_labels = set(df["class_label"].unique()) - set(label2id)

        if unknown_labels:
            raise ValueError(
                f"Unknown categories in {name} data: {unknown_labels}"
            )

    # Make copies so the original DataFrames aren't modified directly.
    train_df = train_df.copy()
    dev_df = dev_df.copy()
    test_df = test_df.copy()

    # Add a numerical "label" column to each dataset.
    # All three datasets use exactly the same mapping.
    for df in (train_df, dev_df, test_df):
        df["label"] = df["class_label"].map(label2id)

    # Return the updated datasets and both label mappings.
    return train_df, dev_df, test_df, label2id, id2label


class CrisisDataset(Dataset):
    """
    Store tokenized text and numerical labels in a format PyTorch can use.

    Each item represents one social media post and its correct category.
    """

    def __init__(self, encodings, labels):
        # encodings contains BERT inputs, including token IDs and attention masks.
        self.encodings = encodings

        # labels contains the numerical category for each post.
        self.labels = labels

    def __len__(self):
        # Tell PyTorch how many posts are in this dataset.
        return len(self.labels)

    def __getitem__(self, index):
        # Get the tokenized inputs for one post.
        # Convert each input into a PyTorch tensor.
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }

        # Add the correct category as a numerical label.
        item["labels"] = torch.tensor(
            int(self.labels[index]), dtype=torch.long
        )

        return item


def tokenize_data(train_df, dev_df, test_df, tokenizer):
    """
    Convert the text into BERT tokens and create PyTorch datasets.

    The tokenizer converts words into numerical token IDs.
    Padding makes sequences the same length, while truncation
    shortens posts that exceed the maximum token length.
    """

    # Start with a maximum sequence length of 128 tokens.
    max_length = 128
    datasets = []

    for df in (train_df, dev_df, test_df):

        # Convert the text column into a list of strings and tokenize it.
        encodings = tokenizer(
            df["text"].astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        # Combine the tokenized text with its correct numerical labels.
        dataset = CrisisDataset(
            encodings,
            df["label"].tolist(),
        )

        datasets.append(dataset)

    return tuple(datasets)


def train_model(train_dataset, dev_dataset):
    """
    Fine-tune BERT using training data and monitor it on validation data.

    This function will be implemented in the next stage.
    It will handle model training and validation.
    """
    raise NotImplementedError("Training will be implemented next.")


def predict(model, dataset):
    """
    Generate category predictions for posts using the trained model.

    This function will be implemented after training is ready.
    """
    raise NotImplementedError("Prediction will be implemented later.")


def save_predictions(predictions, output_path):
    """
    Save model predictions to a file for evaluation and comparison.

    This function will be implemented after prediction is ready.
    """
    raise NotImplementedError("Prediction saving will be implemented later.")
