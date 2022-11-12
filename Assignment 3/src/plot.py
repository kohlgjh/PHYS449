'''Functions for plotting results'''
import matplotlib.pyplot as plt
import os

def generate_loss_report(obj_vals, cross_vals, params):
    '''Plots loss of training vs testing'''
    plt.plot(obj_vals, 'r', label='Training Loss')
    plt.plot(cross_vals, 'b', label="Test Loss")
    plt.plot()
    plt.ylim(0,22)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title('Model Training and Testing Loss')
    plt.xlabel('Epoch x Batch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(os.getcwd(),'plots/loss.png'), dpi=150)