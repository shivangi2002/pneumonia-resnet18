import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses, save_path):
   plt.plot(train_losses, label='Train Loss')
   plt.plot(val_losses, label='Validation Loss')
   plt.legend()
   plt.title('Loss Curve')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.savefig(save_path)
   plt.close()

def plot_confusion_matrix(cm, save_path, class_names=["NORMAL", "PNEUMONIA"]):
    plt.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()