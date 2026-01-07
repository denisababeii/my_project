import os
import matplotlib.pyplot as plt
import torch
import typer

from my_project.data import corrupt_mnist
from my_project.model import MyAwesomeModel

import hydra
import logging
log = logging.getLogger(__name__)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_name="training_conf.yaml", config_path=f"{os.getcwd()}/configs")
def main(cfg) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{cfg.hyperparameters.lr}, {cfg.hyperparameters.batch_size}, {cfg.hyperparameters.epochs}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), f"{os.getcwd()}/../../../models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{os.getcwd()}/../../../reports/figures/training_statistics.png")


if __name__ == "__main__":
    main()
