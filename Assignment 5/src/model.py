import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    '''Encoder portion of the variational auto encoder model'''
    def __init__(self, latent_size, kernal_size) -> None:
        super().__init__()
        self.con_layer1 = nn.Conv2d(1, 7, kernal_size, padding=1, stride=2) # we only have one input channel (i.e. non-RGB)
        self.batch = nn.BatchNorm2d(7)
        self.con_layer2 = nn.Conv2d(7, 14, kernal_size, padding=0, stride=2)
        self.lin1 = nn.Linear(3*3*14, 49) # 14 channels of 3x3 after convolutions
        self.lin_mu = nn.Linear(49, latent_size) # layer for mu
        self.lin_sigma = nn.Linear(49, latent_size) # layer for sigma

        # initialize KL for now
        self.kl = 0

        # setting up the normal distribution generator
        self.normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to("cuda"), torch.tensor([1.0]).to("cuda"))

    def forward(self, x):
        # first convolution layer
        x = self.con_layer1(x)
        x = F.relu(x)

        # batch layer
        x = self.batch(x)
        x = F.relu(x)

        # second convolution layer
        x = self.con_layer2(x)
        x = F.relu(x)

        # flatten and send to lienar layers
        x = torch.flatten(x, start_dim=1)
        x = self.lin1(x)
        x = F.relu(x)

        # separate layers for mu and sigma
        mu = self.lin_mu(x)
        log_sigma = self.lin_sigma(x)
        sigma = torch.exp(log_sigma)

        # generate samples from normal distribution
        sample = self.normal.sample(mu.shape)
        sample = torch.reshape(sample, (sample.shape[0], 20))

        # print("mu shape:", mu.shape)
        # print("log_sigma shape:", log_sigma.shape)
        # print("sigma shape:", sigma.shape)
        # print("sample shape:", sample.shape)
        # combine sigma and mu to produce z
        z = sigma*sample + mu
        
        # KL metric
        self.kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - sigma.pow(2))

        return z

class Decoder(nn.Module):
    def __init__(self, latent_size, kernal_size) -> None:
        super().__init__()
        # reverse the encoder basically

        # linear portion
        self.lin1 = nn.Linear(latent_size, 49)
        self.lin2 = nn.Linear(49, 3*3*14)

        # reshape to 2D
        self.reshape2D = nn.Unflatten(dim=1, unflattened_size=(14, 3, 3)) # unflatten to 14 channels of 3x3

        # convolutional portion
        self.deconv1 = nn.ConvTranspose2d(14, 7, kernal_size, stride=2, output_padding=0)
        self.batch = nn.BatchNorm2d(7)
        self.deconv2 = nn.ConvTranspose2d(7, 1, kernal_size, padding=1, stride=2, output_padding=1)


    def forward(self, z):
        # linear portion of decoding - mirrors encoding
        z = self.lin1(z)
        z = F.relu(z)

        z = self.lin2(z)
        z = F.relu(z)

        # reshape things for convolution
        z = self.reshape2D(z)

        # convolution in reverse
        z = self.deconv1(z)
        z = self.batch(z)
        z = F.relu(z)
        z = self.deconv2(z)

        # output needs to be greater than 1 so use sigmoid
        z = torch.sigmoid(z)

        return z

class VAE(nn.Module):
    '''Class of the variational auto encoder'''
    def __init__(self, latent_size, kernal_size) -> None:
        super().__init__()
        
        self.enc = Encoder(latent_size, kernal_size)
        self.dec = Decoder(latent_size, kernal_size)

    def forward(self, x):
        z = self.enc.forward(x)
        output = self.dec.forward(z)
        return output