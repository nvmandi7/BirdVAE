import torch
from torch import nn

# Model
class BirbSimpleAE(nn.Module):
    def __init__(self, input_size):
        super(BirbSimpleAE, self).__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_size, out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=128)
        
        self.decoder_hidden_layer = nn.Linear(in_features=128, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features=input_size)

    def forward(self, features):
        e1 = self.encoder_hidden_layer(features)
        e1 = torch.relu(e1)
        e2 = self.encoder_output_layer(e1)
        latent_representation = torch.relu(e2)

        d1 = self.decoder_hidden_layer(latent_representation)
        d1 = torch.relu(d1)
        d2 = self.decoder_output_layer(d1)
        reconstructed = torch.relu(d2)
        return reconstructed

    # Probably breaking something by calling this directly, note this if getting issues
    def encode(self, features):
        e1 = self.encoder_hidden_layer(features)
        e1 = torch.relu(e1)
        e2 = self.encoder_output_layer(e1)
        latent_representation = torch.relu(e2)
        return latent_representation

    def decode(self, latent_rep):
        d1 = self.decoder_hidden_layer(latent_rep)
        d1 = torch.relu(d1)
        d2 = self.decoder_output_layer(d1)
        reconstructed = torch.relu(d2)
        return reconstructed


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device\n")
    
    model = BirbSimpleAE(input_size=784).to(device)
    print(model, "\n")

    X = torch.rand(784, device=device)
    reconstructed = model(X)

    print(f"Difference between input and output: {reconstructed - X}")



