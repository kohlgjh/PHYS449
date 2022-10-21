'''Functions to handle the generating of the report on the model'''
import matplotlib.pyplot as plt
import os


def generate_report(obj_vals, cross_vals, accuracy, cwd, num_epochs):
    '''Function to generate a plot roport of the model training'''
    plt.plot(range(num_epochs), obj_vals, 'r', label='Training Loss')
    plt.plot(range(num_epochs), cross_vals, 'b', label="Test Loss")
    plt.plot()
    plt.ylim(0,1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title('MNIST Even Numbers -\nModel Training and Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(cwd,'model_report.pdf'), dpi=150)
    plt.show()