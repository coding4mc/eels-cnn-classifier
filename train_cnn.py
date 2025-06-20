from lib.log import setup_logging
from lib.func import load_training_data
from lib.ml import CNN, CNN2, CNN3, CNN4, CNN5, CNN6, ModelTrainer
import torch
import os
import logging

Model = CNN6

training_data_file_name = "250413_Ni_training-data-reduced-1000-rescaled-augmented-20x-noise.pkl"
training_data_base_name, _ = os.path.splitext(training_data_file_name)
setup_logging(log_file_path=f"logs/{training_data_base_name}-{Model.__name__.lower()}.log")
log = logging.getLogger(__name__)
    
data = load_training_data(f"Training Data/{training_data_file_name}")
train_data, test_data = data.split_train_test()

device = torch.device('mps')

model = Model(
    spectra_data_point_count=data.spectra_data_point_count,
    unique_label_count=data.unique_label_count,
    device=device
)

model_trainer = ModelTrainer(model=model, device=device)

num_epochs = 10
for i in range(1, num_epochs + 1):
    model_trainer.train(train_data=train_data, test_data=test_data, batch_size=4096, num_epochs=1)

    model_file_name = f"{training_data_base_name}-{model.__class__.__name__.lower()}-{i}epoch.torch"
    model_file_path = f"Trained CNN Model/{model_file_name}"
    torch.save(model.state_dict(), model_file_path)

    log.info(f"Model saved: {model_file_name}")