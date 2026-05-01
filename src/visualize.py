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
