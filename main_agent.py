import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentException(Exception):
    """Base exception class for agent-related errors."""
    pass

class InvalidConfigurationException(AgentException):
    """Raised when the configuration is invalid."""
    pass

class Agent:
    """
    Main agent implementation.

    This class represents the main agent in the system, responsible for
    interacting with the environment and making decisions based on the
    current state.

    Attributes:
        config (Dict): Configuration dictionary.
        data (pd.DataFrame): Dataframe containing the data.
        model (torch.nn.Module): PyTorch model instance.
    """

    def __init__(self, config: Dict):
        """
        Initializes the agent with the given configuration.

        Args:
            config (Dict): Configuration dictionary.

        Raises:
            InvalidConfigurationException: If the configuration is invalid.
        """
        self.config = config
        self.data = None
        self.model = None

        # Validate configuration
        if not self._validate_config():
            raise InvalidConfigurationException("Invalid configuration")

    def _validate_config(self) -> bool:
        """
        Validates the configuration.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        # Check if required keys are present
        required_keys = ["data_path", "model_type"]
        for key in required_keys:
            if key not in self.config:
                logging.error(f"Missing required key: {key}")
                return False

        # Check if data path exists
        if not self.config["data_path"]:
            logging.error("Data path is empty")
            return False

        return True

    def load_data(self) -> None:
        """
        Loads the data from the specified path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            self.data = pd.read_csv(self.config["data_path"])
            logging.info(f"Loaded data from {self.config['data_path']}")
        except FileNotFoundError:
            logging.error(f"File not found: {self.config['data_path']}")
            raise

    def create_model(self) -> None:
        """
        Creates the PyTorch model instance.

        Raises:
            ValueError: If the model type is invalid.
        """
        if self.config["model_type"] == "linear":
            self.model = torch.nn.Linear(1, 1)
            logging.info("Created linear model")
        else:
            logging.error(f"Invalid model type: {self.config['model_type']}")
            raise ValueError("Invalid model type")

    def train_model(self) -> None:
        """
        Trains the model using the loaded data.

        Raises:
            ValueError: If the data is not loaded.
        """
        if self.data is None:
            logging.error("Data is not loaded")
            raise ValueError("Data is not loaded")

        # Define loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        # Train model
        for epoch in range(100):
            # Forward pass
            inputs = torch.tensor(self.data["input"].values, dtype=torch.float32)
            labels = torch.tensor(self.data["label"].values, dtype=torch.float32)
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def evaluate_model(self) -> float:
        """
        Evaluates the model using the loaded data.

        Returns:
            float: Evaluation metric (MSE).

        Raises:
            ValueError: If the data is not loaded.
        """
        if self.data is None:
            logging.error("Data is not loaded")
            raise ValueError("Data is not loaded")

        # Define loss function
        criterion = torch.nn.MSELoss()

        # Evaluate model
        inputs = torch.tensor(self.data["input"].values, dtype=torch.float32)
        labels = torch.tensor(self.data["label"].values, dtype=torch.float32)
        outputs = self.model(inputs)
        loss = criterion(outputs, labels)

        logging.info(f"Evaluation Metric (MSE): {loss.item()}")

        return loss.item()

    def save_model(self) -> None:
        """
        Saves the trained model to a file.

        Raises:
            ValueError: If the model is not trained.
        """
        if self.model is None:
            logging.error("Model is not trained")
            raise ValueError("Model is not trained")

        torch.save(self.model.state_dict(), "model.pth")
        logging.info("Saved model to model.pth")

    def load_model(self) -> None:
        """
        Loads the trained model from a file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        try:
            self.model.load_state_dict(torch.load("model.pth"))
            logging.info("Loaded model from model.pth")
        except FileNotFoundError:
            logging.error("File not found: model.pth")
            raise

def main():
    # Create configuration dictionary
    config = {
        "data_path": "data.csv",
        "model_type": "linear"
    }

    # Create agent instance
    agent = Agent(config)

    # Load data
    agent.load_data()

    # Create model
    agent.create_model()

    # Train model
    agent.train_model()

    # Evaluate model
    evaluation_metric = agent.evaluate_model()

    # Save model
    agent.save_model()

    # Load model
    agent.load_model()

if __name__ == "__main__":
    main()