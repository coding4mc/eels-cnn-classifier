from typing import Dict
import torch
import datetime as dt
from torch import nn, optim

from lib.model import TrainingData, ValidationOutput, Accuracy
from lib.log import Logger


class ModelTrainer(Logger):
    """
    A class to train a machine learning model using a specified dataset and device.

    Attributes:
        _model (nn.Module): The machine learning model to be trained.
        _device (torch.device): The device (CPU or GPU) on which to train the model.
        _criterion (nn.CrossEntropyLoss): The loss function used for training.
        _optimizer (optim.Adam): The optimizer used for training.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the MlTrainer with a model and device.

        Args:
            model (nn.Module): The machine learning model to be trained.
            device (torch.device): The device (CPU or GPU) on which to train the model.
        """
        super().__init__()
        self._model = model
        self._device = device
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

    def train(
            self,
            train_data: TrainingData,
            test_data: TrainingData,
            batch_size: int,
            num_epochs: int
    ) -> ValidationOutput:
        """
        Train the model using the specified dataset, batch size, and number of epochs.

        Args:
            dataset (ModelDataset): The dataset to be used for training.
            batch_size (int): The batch size for training.
            num_epochs (int): The number of epochs for training.
        """
        train_data_loader = train_data.get_data_loader(batch_size=batch_size, device=self._device)
        self._log.info(f"Started training")
        for epoch in range(num_epochs):
            self._model.train()

            validation = None
            processed_count = 0
            next_log_threshold = train_data.sample_count // 10
            self._log.info(f"Epoch {epoch + 1}")
            for epoch_inputs, epoch_train_classes in train_data_loader:
                epoch_inputs, epoch_train_classes = epoch_inputs.to(self._device), epoch_train_classes.to(self._device)
                epoch_inputs: torch.Tensor
                epoch_train_classes: torch.Tensor

                # Run model
                self._optimizer.zero_grad()
                outputs = self._model(epoch_inputs)
                loss = self._criterion(outputs, epoch_train_classes)
                loss.backward()
                self._optimizer.step()

                # Calculate validation statistics
                _, predicted_classes = outputs.max(1)
                epoch_validation = self._build_validation_output(
                    loss=loss,
                    predicted_classes=predicted_classes,
                    training_classes=epoch_train_classes,
                    classification_to_label_map=train_data.classification_to_label_map
                )

                validation = (
                    validation.merge_with(epoch_validation)
                    if validation
                    else epoch_validation
                )

                # Log epoch
                processed_count += epoch_train_classes.shape[0]
                if processed_count >= next_log_threshold:
                    self._log.info(f"[{epoch + 1}/{num_epochs}]: Processed {processed_count} / {train_data.sample_count} samples")
                    next_log_threshold += train_data.sample_count // 10

            self._log.info(f'[{epoch + 1}/{num_epochs}]: Validation {validation}')

        # Validation
        test_validation = self.validate(test_data=test_data, batch_size=batch_size)
        self._log.info(f'Validation - {test_validation}')
        return test_validation
        

    def validate(
            self,
            test_data: TrainingData,
            batch_size: int,
    ) -> ValidationOutput:
        self._model.eval()
        
        test_validation = None
        test_data_loader = test_data.get_data_loader(batch_size=batch_size, device=self._device)
        with torch.no_grad():
            for test_inputs, test_classes in test_data_loader:
                batch_validation = self._validate_batch(
                    input_tensor=test_inputs,
                    input_classes=test_classes,
                    classification_to_label_map=test_data.classification_to_label_map
                )
                test_validation = (
                    test_validation.merge_with(batch_validation)
                    if test_validation
                    else batch_validation
                )

        return test_validation

    def _validate_batch(
            self,
            input_tensor: torch.Tensor,
            input_classes: torch.Tensor,
            classification_to_label_map: Dict[int, int]
    ) -> ValidationOutput:
        """
        Validate the model on a given input tensor and labels.

        Args:
            input_tensor (torch.Tensor): The input tensor for validation.
            labels (torch.Tensor): The labels for validation.

        Returns:
            ValidationOutput: The validation output containing loss, correct count, and total count.
        """
        input_tensor = input_tensor.to(self._device)
        input_classes = input_classes.to(self._device)

        outputs = self._model(input_tensor)
        loss = self._criterion(outputs, input_classes)
        _, predicted_classes = outputs.max(1)
        
        return self._build_validation_output(
            loss=loss,
            predicted_classes=predicted_classes,
            training_classes=input_classes,
            classification_to_label_map=classification_to_label_map
        )
    
    def _build_validation_output(
            self,
            loss: torch.Tensor,
            predicted_classes: torch.Tensor,
            training_classes: torch.Tensor,
            classification_to_label_map: Dict[int, int]
    ):
        overall_total_count = training_classes.size(0)
        overall_correct_count = predicted_classes.eq(training_classes).sum().item()

        return ValidationOutput(
            loss=loss.item(),
            overall_accuracy=Accuracy(
                correct_count=overall_correct_count,
                total_count=overall_total_count
            ),
            accuracy_per_label=self._calculate_accuracy_per_label(
                predicted_classes=predicted_classes,
                training_classes=training_classes,
                classification_to_label_map=classification_to_label_map
            )
        )

    def _calculate_accuracy_per_label(
            self,
            predicted_classes: torch.Tensor,
            training_classes: torch.Tensor,
            classification_to_label_map: Dict[int, int]
    ):
        accuracy_per_label: Dict[int, Accuracy] = {}

        # Loop through each class
        for training_class_value in training_classes.unique():
            training_class_value: torch.Tensor

            # Calculate the correct/total count for the class.
            # We filter the predictions down to the ones where the correct answer is the class.
            # Therefore, we can calculate the total count (all predictions that should equal the class)
            # and the correct count (all predictions that should and is equal to the class).
            indexes = torch.argwhere(training_classes == training_class_value).reshape(-1)
            correct_count = predicted_classes[indexes].eq(training_classes[indexes]).sum().item()
            total_count = training_classes.eq(training_class_value).sum().item()

            # Update the dictionary.
            accuracy = Accuracy(correct_count=correct_count, total_count=total_count)
            training_label = classification_to_label_map[training_class_value.item()]
            accuracy_per_label[training_label] = accuracy

        return accuracy_per_label