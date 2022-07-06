
import os, sys
sys.path.append("/Users/nathanmandi/ML-stuff/BirdVAE/modeling")

import torch, torchvision
from torchvision import transforms, datasets
from torch import nn, optim
from dataset import Birds400Dataset
from modeling.birdvae import BirdVAE, vae_loss_function


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdVAE(input_shape=(3, 240, 240), embedding_size=784).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = vae_loss_function


    # Datasets
    
    # TODO transform in various ways: look up common transforms. Include normalization. Batch normalize
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    data_dir = "/Users/nathanmandi/ML-stuff/BirdVAE/data/birds400"
    train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)
    test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False, num_workers=4
    )


    print(f"Using {device} device")
    print(model)
    # print(f"Loss function: {loss_fn}")
    # print(f"Train / Test Data Points: {len(train_subset)} / {len(test_subset)}\n")

    epochs = 10
    for epoch in range(epochs):
        size = len(train_loader.dataset)
        loss = 0
        for step, (X, y) in enumerate(train_loader):
            # One image: 3 x 240 x 240 = 172,800
            # reshape mini-batch data to [N, 172,800] matrix, load it to the active device
            X = X.view(-1, 172800).to(device)

            optimizer.zero_grad()
            reconstructed, mu, logvar = model(X)
            
            # Reconstruction and divergence losses
            train_loss = criterion(reconstructed, X, mu, logvar)
            train_loss.backward()

            optimizer.step()
            loss += train_loss.item()

            if step % 100 == 0:
                loss, current = loss, step * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # compute the epoch training loss
        loss = loss / len(train_loader)
        
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
    

    model_output_path = "/Users/nathanmandi/ML-stuff/BirdVAE/models/test_model.pth"
    torch.save(model.state_dict(), open(model_output_path, 'wb'))
    print(f"Saved model to {model_output_path}")
