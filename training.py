
import os
import torch, torchvision
from torch import nn, optim
# from dataset import Birds400Dataset
from modeling.ae import BirbSimpleAE
from modeling.birdvae import BirdVAE, vae_loss_function

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdVAE(input_size=784).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = vae_loss_function


    # Datasets
    
    # TODO transform in various ways: look up common transforms. Include normalization. Batch normalize
    transform = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # train_dataset = Birds400Dataset(train=True)
    # test_dataset = Birds400Dataset(train=False)

    data_dir = "/Users/nathanmandi/ML-stuff/BirdVAE/data/birds400"
    train_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)
    test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False, num_workers=4
    )


    print(f"Using {device} device")
    print(model)
    # print(f"Loss function: {loss_fn}")
    # print(f"Optimizer: {HPARAMS['optimizer']}")
    # print(f"Train / Test Data Points: {len(train_subset)} / {len(test_subset)}\n")

    epochs = 10
    for epoch in range(epochs):
        size = len(train_loader.dataset)
        loss = 0
        for step, (X, y) in enumerate(train_loader):
            # reshape mini-batch data to [N, 784] matrix, load it to the active device
            X = X.view(-1, 784).to(device)

            optimizer.zero_grad()
            outputs = model(X)
            
            # Reconstruction loss
            train_loss = criterion(outputs, X)
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
