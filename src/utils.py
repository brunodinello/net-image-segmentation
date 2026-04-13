import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report


def dice_coefficient(
    preds: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Compute the Dice coefficient between predictions and targets.

    Parameters
    ----------
    preds : torch.Tensor
        Raw model output (logits). Sigmoid activation is applied internally.
    targets : torch.Tensor
        Ground-truth binary masks with the same shape as ``preds``.
    smooth : float, optional
        Smoothing constant to avoid division by zero, by default 1e-6.

    Returns
    -------
    torch.Tensor
        Scalar Dice coefficient in the range [0, 1].
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice


class DiceLoss(nn.Module):
    """Differentiable Dice loss for binary segmentation.

    Parameters
    ----------
    smooth : float, optional
        Smoothing constant to avoid division by zero, by default 1e-6.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        preds : torch.Tensor
            Raw model output (logits).
        targets : torch.Tensor
            Ground-truth binary masks.

        Returns
        -------
        torch.Tensor
            Scalar loss value: ``1 - Dice score``.
        """
        preds = torch.sigmoid(preds)

        intersection = (preds * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice_score


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate a model on a dataset and return average loss and Dice score.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    criterion : nn.Module
        Loss function used to compute per-batch loss.
    data_loader : torch.utils.data.DataLoader
        DataLoader providing the evaluation batches.
    device : torch.device
        Device on which to run inference.

    Returns
    -------
    avg_loss : float
        Mean loss across all batches.
    avg_dice : float
        Mean Dice coefficient across all batches.
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).float()
            output = model(x)
            total_loss += criterion(output, y).item()
            total_dice += dice_coefficient(output, y).item()

    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)

    return avg_loss, avg_dice


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait after the last improvement before
        stopping, by default 5.
    """

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.val_loss_min = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Update the stopping state based on the current validation loss.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.
        """
        if val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def print_log(
    epoch: int,
    train_loss: float,
    val_loss: float,
    train_dice: float,
    val_dice: float,
) -> None:
    """Print a formatted summary line for the current training epoch.

    Parameters
    ----------
    epoch : int
        Zero-based epoch index.
    train_loss : float
        Average training loss for the epoch.
    val_loss : float
        Average validation loss for the epoch.
    train_dice : float
        Average training Dice coefficient for the epoch.
    val_dice : float
        Average validation Dice coefficient for the epoch.
    """
    print(
        f"Epoch: {epoch + 1:03d} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"Train Dice: {train_dice:.5f} | "
        f"Val Dice: {val_dice:.5f}"
    )


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    do_early_stopping: bool = True,
    patience: int = 5,
    epochs: int = 10,
    log_fn: Optional[Callable] = print_log,
    log_every: int = 1,
    scheduler: Optional[object] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a segmentation model and track loss and Dice score per epoch.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    criterion : nn.Module
        Loss function used to compute per-batch loss.
    train_loader : torch.utils.data.DataLoader
        DataLoader providing training batches.
    val_loader : torch.utils.data.DataLoader
        DataLoader providing validation batches.
    device : torch.device
        Device on which to run training.
    do_early_stopping : bool, optional
        Whether to enable early stopping, by default True.
    patience : int, optional
        Early stopping patience in epochs, by default 5.
    epochs : int, optional
        Maximum number of training epochs, by default 10.
    log_fn : callable, optional
        Function called after each ``log_every`` epochs with signature
        ``(epoch, train_loss, val_loss, train_dice, val_dice)``,
        by default :func:`print_log`.
    log_every : int, optional
        Logging frequency in epochs, by default 1.
    scheduler : optional
        Learning rate scheduler with a ``step(metric)`` interface,
        by default None.

    Returns
    -------
    epoch_train_losses : list of float
    epoch_val_losses : list of float
    epoch_train_dice : list of float
    epoch_val_dice : list of float
    """
    epoch_train_losses: List[float] = []
    epoch_val_losses: List[float] = []
    epoch_train_dice: List[float] = []
    epoch_val_dice: List[float] = []

    if do_early_stopping:
        early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            output = model(x)
            batch_loss = criterion(output, y)
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()
            train_dice += dice_coefficient(output, y).item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        epoch_train_losses.append(train_loss)
        epoch_train_dice.append(train_dice)

        val_loss, val_dice = evaluate(model, criterion, val_loader, device)
        epoch_val_losses.append(val_loss)
        epoch_val_dice.append(val_dice)

        if scheduler is not None:
            scheduler.step(val_dice)

        if do_early_stopping:
            early_stopping(val_loss)

        if log_fn is not None and (epoch + 1) % log_every == 0:
            log_fn(epoch, train_loss, val_loss, train_dice, val_dice)

        if do_early_stopping and early_stopping.early_stop:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best validation loss: {early_stopping.best_score:.5f}"
            )
            break

    return epoch_train_losses, epoch_val_losses, epoch_train_dice, epoch_val_dice


def plot_training(
    train_losses: List[float],
    val_losses: List[float],
    train_dice: List[float],
    val_dice: List[float],
) -> None:
    """Plot training and validation loss and Dice score curves side by side.

    Parameters
    ----------
    train_losses : list of float
        Training loss per epoch.
    val_losses : list of float
        Validation loss per epoch.
    train_dice : list of float
        Training Dice coefficient per epoch.
    val_dice : list of float
        Validation Dice coefficient per epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    tick_step = 10
    ticks = [1] + list(range(10, len(train_losses) + 1, tick_step))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Training", linewidth=2, marker="o", markersize=3)
    axes[0].plot(epochs, val_losses, label="Validation", linewidth=2, marker="s", markersize=3)
    axes[0].set_title("Training and validation loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_xticks(ticks)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_dice, label="Training", linewidth=2, marker="o", markersize=3)
    axes[1].plot(epochs, val_dice, label="Validation", linewidth=2, marker="s", markersize=3)
    axes[1].set_title("Training and validation Dice score", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Dice score", fontsize=12)
    axes[1].set_xticks(ticks)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.show()


def model_classification_report(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    nclasses: int,
) -> None:
    """Print accuracy and a per-class classification report for a model.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing evaluation batches.
    device : torch.device
        Device on which to run inference.
    nclasses : int
        Number of target classes, used to label the classification report.
    """
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(nclasses)]
    )
    print("Classification report:\n", report)


def show_tensor_image(
    tensor: torch.Tensor,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Display a single image tensor using Matplotlib.

    Parameters
    ----------
    tensor : torch.Tensor
        Image tensor of shape ``(C, H, W)``. Grayscale (C=1) and RGB (C=3)
        are both supported.
    title : str, optional
        Title displayed above the image, by default None.
    vmin : float, optional
        Minimum value for the colormap scale, by default None.
    vmax : float, optional
        Maximum value for the colormap scale, by default None.
    """
    if tensor.shape[0] == 1:
        plt.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def show_tensor_images(
    tensors: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Display a list of image tensors in a single row using Matplotlib.

    Parameters
    ----------
    tensors : list of torch.Tensor
        List of image tensors, each of shape ``(C, H, W)``.
    titles : list of str, optional
        Titles for each image. Must match the length of ``tensors`` if
        provided, by default None.
    figsize : tuple of int, optional
        Figure size as ``(width, height)`` in inches, by default (15, 5).
    vmin : float, optional
        Minimum value for the colormap scale, by default None.
    vmax : float, optional
        Maximum value for the colormap scale, by default None.
    """
    num_images = len(tensors)
    _, axs = plt.subplots(1, num_images, figsize=figsize)
    for i, tensor in enumerate(tensors):
        ax = axs[i]
        if tensor.shape[0] == 1:
            ax.imshow(tensor.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(tensor.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        if titles and titles[i]:
            ax.set_title(titles[i])
        ax.axis("off")
    plt.show()
