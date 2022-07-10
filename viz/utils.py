
from torch.utils.data import Dataset

def plot_random_datapoints(dataset: Dataset, rows=3, cols=3):
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze().permute(1, 2, 0), cmap="gray")
    plt.show()
    return

