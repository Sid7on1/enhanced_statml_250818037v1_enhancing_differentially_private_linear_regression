import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for CUDA availability
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
logger.info("Device in use: {}".format(device))

# Define global constants
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "models")
TENSORBOARD_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Create directories if they don't exist
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(TENSORBOARD_LOG_DIR):
    os.makedirs(TENSORBOARD_LOG_DIR)

# Define exception classes
class ModelTrainingError(Exception):
    """Custom exception class for errors during model training."""

class InvalidDatasetError(Exception):
    """Custom exception class for invalid dataset errors."""

# Helper functions
def load_dataset(data_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess the dataset.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Preprocessed features and targets.
    """
    # TODO: Implement dataset loading and preprocessing
    # Return features and targets as torch tensors
    raise NotImplementedError("Dataset loading and preprocessing logic needs to be implemented.")

def train_model(model: nn.Module,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                epochs: int,
                learning_rate: float,
                weight_decay: float,
                patience: int,
                device: str) -> nn.Module:
    """
    Train the model using the provided data loaders.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Data loader for training data.
        valid_loader (DataLoader): Data loader for validation data.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) parameter.
        patience (int): Early stopping patience (number of epochs).
        device (str): Device to use for training ("cpu" or "cuda").

    Returns:
        nn.Module: Trained model.

    Raises:
        ModelTrainingError: If there is an error during model training.
    """
    logger.info("Starting model training...")

    # Move model to the specified device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Set up early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize tensorboard writer
    current_time = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_LOG_DIR, current_time))

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate validation loss
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for features, targets in valid_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()

        # Average losses across batches
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        # Log losses and write to tensorboard
        logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} - Valid Loss: {avg_valid_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch+1)
        writer.add_scalar("Loss/Validation", avg_valid_loss, epoch+1)

        # Early stopping
        early_stopping(avg_valid_loss, model, model_path=os.path.join(MODEL_SAVE_PATH, "early_stopped_model.pt"))
        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Stopping training...")
            break

    writer.close()

    # Load the best model weights
    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "early_stopped_model.pt")))

    return model

# Early stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                        Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                        Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                        Default: 0
            path (str): Path for the checkpoint to be saved to.
                        Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                        Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model, model_path=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_path=None):
        '''Saves model when validation loss decrease.'''
        if model_path is None:
            model_path = self.path
        torch.save(model.state_dict(), model_path)
        self.val_loss_min = val_loss

# Main training class
class AgentTrainer:
    """Agent training pipeline."""
    def __init__(self, config: dict):
        """
        Initialize the AgentTrainer.

        Args:
            config (dict): Configuration settings for the trainer.
        """
        self.config = config
        self.model = None
        self.device = config["device"]
        self.dataset_path = config["dataset_path"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.patience = config["patience"]
        self.seed = config["seed"]

    def load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed features and targets.
        """
        try:
            logger.info("Loading and preprocessing dataset...")
            features, targets = load_dataset(self.dataset_path)

            # Shuffle the data
            indices = torch.randperm(features.size(0))
            features = features[indices]
            targets = targets[indices]

            return features, targets
        except Exception as e:
            logger.error("Error loading dataset: {}".format(str(e)))
            raise InvalidDatasetError("Failed to load dataset.")

    def create_data_loaders(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.

        Args:
            features (torch.Tensor): Input features.
            targets (torch.Tensor): Target values.

        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders.
        """
        # Split the data into training and validation sets
        train_size = int(0.8 * features.size(0))
        valid_size = features.size(0) - train_size
        train_features, valid_features = features[:train_size], features[train_size:]
        train_targets, valid_targets = targets[:train_size], targets[train_size:]

        # Create data loaders
        train_loader = DataLoader(list(zip(train_features, train_targets)), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(list(zip(valid_features, valid_targets)), batch_size=self.batch_size, shuffle=False)

        return train_loader, valid_loader

    def build_model(self) -> nn.Module:
        """
        Build and return the agent model.

        Returns:
            nn.Module: The agent model.
        """
        # TODO: Implement model architecture
        # Return the built model
        raise NotImplementedError("Model architecture needs to be implemented.")

    def train(self) -> None:
        """Train the agent model."""
        try:
            logger.info("Starting agent training...")

            # Set random seed for reproducibility
            torch.manual_seed(self.seed)

            # Load and preprocess the dataset
            features, targets = self.load_dataset()

            # Create data loaders
            train_loader, valid_loader = self.create_data_loaders(features, targets)

            # Build the model
            self.model = self.build_model()

            # Train the model
            self.model = train_model(self.model, train_loader, valid_loader, self.epochs,
                                    self.learning_rate, self.weight_decay, self.patience, self.device)

            logger.info("Agent training completed.")
        except Exception as e:
            logger.error("Error during agent training: {}".format(str(e)))
            raise ModelTrainingError("Failed to train the agent model.")

    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        Save the trained model to a file.

        Args:
            model_path (Optional[str], optional): Path to save the model. If None, uses the default path. Defaults to None.
        """
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_PATH, "final_model.pt")

        torch.save(self.model.state_dict(), model_path)
        logger.info("Model saved to: {}".format(model_path))

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        "device": device,
        "dataset_path": "path/to/your/dataset.csv",
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 100,
        "patience": 10,
        "seed": 42
    }

    # Create trainer instance
    trainer = AgentTrainer(config)

    # Train the agent
    trainer.train()

    # Save the trained model
    trainer.save_model()