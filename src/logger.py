import csv
import json
import os


def save_run_summary(hyperparameters,augmented, best_epoch, history, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(
        csv_path,
        mode="a",
        newline=""
    ) as file:
        writer = csv.writer(file)
        if not file_exists:
             writer.writerow(
                ["LR", "Batch Size", "Num Epochs","Augmented","Best Epoch","Stop Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Val Precision", "Val Recall"]
            )
        
        writer.writerow(
            [hyperparameters["lr"], hyperparameters["images_per_batch"], hyperparameters["num_epochs"], augmented, best_epoch, history["stop_epoch"], history["train_loss"][best_epoch], history["val_loss"][best_epoch], history["val_accuracy"][best_epoch], history["val_precision"][best_epoch], history["val_recall"][best_epoch]])

def save_run_history(run_history, save_path):
    with open(save_path, "w") as f:
        json.dump(run_history, f, indent=4)