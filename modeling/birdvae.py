import torch
from torch import nn, optim
from torch.nn import functional as F
from cnn_architectures import CNN_Encoder, CNN_Decoder

# Model
class BirdVAE(nn.Module):
    def __init__(self, input_shape, embedding_size):
        super(BirdVAE, self).__init__()
        self.encoder = CNN_Encoder(input_shape, embedding_size)
        
        self.logvar = nn.Linear(embedding_size, embedding_size)
        self.mu = nn.Linear(embedding_size, embedding_size)

        self.decoder = CNN_Decoder(embedding_size, input_shape)


    def encode(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
    def random_sample(self):
        eps = torch.randn(embedding_size)
        return self.decode(eps)


    def interpolate(self, x1, x2):
        # Encodes x1 and x2, then computes the mean of the distributions, samples from that
        mu1, logvar1 = self.encode(x1)
        mu2, logvar2 = self.encode(x2)

        mu = 0.5 * (mu1 + mu2)
        logvar = torch.log(0.25 * (torch.exp(logvar1) + torch.exp(logvar2)))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

        


    # Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3*240*240), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")
    
    model = BirbVAE(embedding_size=784).to(device)
    print(model, "\n")

    X = torch.rand(784, device=device)
    reconstructed, mu, logvar = model(X)

    print(f"Difference between input and output: {reconstructed - X}")



