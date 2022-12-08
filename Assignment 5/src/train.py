import torch.optim as optim
import torch

def train(vae_model, input_loader, verbose):
    '''Trains the passed model with passed data loader. Returns objective vals'''
    print("Beginning model training...")
    learning_rate = 0.001
    num_epochs = 30
    optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

    # bce = nn.BCELoss(reduction='sum')
    obj_vals = []

    for epoch in range(num_epochs):
        loss_sum = 0
        for (inputs, _) in input_loader:
            loss = torch.sum((inputs - vae_model.forward(inputs)).pow(2)) + vae_model.enc.kl # using MSE loss & KL
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        obj_val = loss_sum/len(input_loader.dataset)
        obj_vals.append(obj_val)

        if (epoch+1) % 5 == 0 and verbose:
            print(f'Epoch {epoch+1}/{num_epochs}: \t Loss: {obj_val:.2f}')

    print(f"Final Loss: \t {obj_val:.2f}")
    print("Model training complete...")

    return obj_vals