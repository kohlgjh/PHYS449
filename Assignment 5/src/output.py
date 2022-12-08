import matplotlib.pyplot as plt
import os
import numpy as np

def output(vae_model, input, obj_vals, n, dir):
    '''Outputs plot of loss, and n digit images to passed dir'''

    # check if directory path exists and make it
    if not os.path.exists(os.path.join(os.getcwd(), dir)):
        os.mkdir(os.path.join(os.getcwd(), dir))

    # produce plot of loss
    plt.style.use('default')
    plt.plot(obj_vals)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(dir, 'loss.pdf'))
    plt.clf()

    # produce plots for n number of digits
    output = vae_model.forward(input)
    output_np = output.detach().cpu().numpy()
    for i in range(int(n)):
        plt.imshow(output_np[i][0], cmap='Greys')
        plt.savefig(os.path.join(dir, f'{i+1}.pdf'))
