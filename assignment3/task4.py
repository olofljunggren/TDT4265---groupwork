import torchvision
from torch import nn
import pathlib
import matplotlib.pyplot as plt
import utils
from dataloaders import load_cifar10
from trainer import compute_loss_and_accuracy#, Trainer

# New imports
import typing
import torch
import time
import collections


class Trainer():

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader]):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                      self.learning_rate,
                                      )

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.TRAIN_ACC = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC[self.global_step] = validation_acc
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f},",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep="\t")
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC[self.global_step] = test_acc
        self.TEST_LOSS[self.global_step] = test_loss
        print(f"Test Loss: {test_loss:.2f},",
        	  f"Test Accuracy: {test_acc:.3f}",sep="\t")

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def calculate_training_loss_acc(self):
        train_toss, train_acc = compute_loss_and_accuracy(self.dataloader_train, self.model, self.loss_criterion)
        print(f'training loss: {train_toss:.2f},', f'training accuracy: {train_acc:.3f}',sep="\t")

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = Model.forward(self, X_batch)

                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()

                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()
                self.global_step += 1
                 # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        self.calculate_training_loss_acc()
                        return
        self.calculate_training_loss_acc()

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,

        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
    
    def forward(self, x):
        x = self.model(x)
        return x
    
def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.TRAIN_LOSS, label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 5
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model()

    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    print("Training begins...")
    trainer.train()
    create_plots(trainer, "Task 4")

    # Get/calculate accuracies
    test_data = trainer.dataloader_test
    train_data = trainer.dataloader_train
    val_data = trainer.dataloader_val

    test_accuracy = compute_loss_and_accuracy(dataloader=test_data, model=model, loss_criterion=trainer.loss_criterion)
    print("Test accuracy:", test_accuracy)

    validation_accuracy = compute_loss_and_accuracy(dataloader=val_data, model=model, loss_criterion=trainer.loss_criterion)
    print("Validation accuracy:", validation_accuracy)

    train_accuracy = compute_loss_and_accuracy(dataloader=train_data, model=model, loss_criterion=trainer.loss_criterion)
    print("Train accuracy:", train_accuracy)


if __name__ == "__main__":
    main()