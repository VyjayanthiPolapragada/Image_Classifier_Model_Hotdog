#importing libraries

from copy import deepcopy
from typing import List, Tuple
from os.path import join
from os import getcwd

from matplotlib.pyplot import subplots, show
from numpy import sum
from torch import argmax, device, cuda, save, load
from torch.nn import Module
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#creating classifier class

class ClassificationModelTrainer:

    def __init__(self,
                 model: Module,
                 training_set: Dataset,
                 validation_set: Dataset,
                 batch_size: int,
                 minimising_criterion: _Loss,
                 optimiser: Optimizer) -> None:
        
        #Initialise a classification model training module.

        # model: The model to train.
        # training_set: The set of training data.
        # validation_set: The set of validation data.
        # batch_size: The batch size for training.
        # minimising_criterion: The loss function.
        # optimiser: The algorithm to perform minimisation task.

        self._device = device("cuda:0" if cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._train_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
        self._validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)
        self._minimising_criterion = minimising_criterion
        self._optimiser = optimiser
        self.training_loss = []
        self.validation_acc = []

    def get_model(self) -> Module:

        #Getter function for model.

        #return the trained model.

        return self._model

    def train_model(self, n_epochs) -> None:

        #Perform the model training.

        # n_epochs: The number of training epochs to run.

        # Setup the progress bar.
        pbar = tqdm(total=n_epochs * (len(self._train_loader) + len(self._validation_loader)))
        pbar.set_postfix({
            "Training Loss": "Not yet available" if len(self.training_loss) == 0 else self.training_loss[-1],
            "Validation Accuracy": "Not yet available" if len(self.validation_acc) == 0 else self.validation_acc[-1],
            "Epoch": 1
        })
        # Training through the epochs.
        for epoch in range(n_epochs):
            loss_sublist = []
            # Training Process
            for x, y in self._train_loader:
                x, y = x.to(self._device), y.to(self._device)
                self._model.train()
                z = self._model(x)
                loss = self._minimising_criterion(z, y)
                loss_sublist.append(loss.data.item())
                loss.backward()
                self._optimiser.step()
                self._optimiser.zero_grad()
                pbar.update()
            self.training_loss.append(sum(loss_sublist))
            # Validation Process
            correct = 0
            n_test = 0
            for x_test, y_test in self._validation_loader:
                x_test, y_test = x_test.to(self._device), y_test.to(self._device)
                self._model.eval()
                z = softmax(self._model(x_test), dim=1)
                y_hat = argmax(z.data, dim=1)
                correct += (y_hat == y_test).sum().item()
                n_test += y_hat.shape[0]
                pbar.update()
            accuracy = correct / n_test
            self.validation_acc.append(accuracy)
            pbar.set_postfix({
                "Training Loss": self.training_loss[-1],
                "Validation Accuracy": accuracy,
                "Epoch": epoch + 2
            })
        pbar.set_postfix({
            "Training Loss": self.training_loss[-1],
            "Validation Accuracy": self.validation_acc[-1],
            "Epoch": n_epochs
        })

    def plot_training_stat(self):

        #This function plots the training statistics the model trainer collected 
        #throughout the training process. Namely, they are

        # Total training loss versus Iterations, and
        # Validation Accuracy versus Iterations.

        #The two statistics are placed in the same plot, respectively in red and blue.

        # Plot Total training loss versus Iterations
        fig, ax1 = subplots()
        color = 'tab:red'
        ax1.plot(self.training_loss, color=color)
        ax1.set_xlabel('Iterations', color="black")
        ax1.set_ylabel('Total Training Loss', color=color)
        ax1.set_ylim(bottom = 0)
        ax1.tick_params(axis='y', color=color)
        # Plot Validation Accuracy versus Iterations
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation Accuracy', color=color)
        ax2.plot(self.validation_acc, color=color)
        ax2.tick_params(axis='y', color=color)
        ax2.set_ylim(0, 1)
        fig.tight_layout()
        show()

    def test(self, testing_data: Dataset) -> float:

        #This function tests the model's performance on a given dataset.

        #testing_data: The dataset to perform testing upon.
        #return: Model's accuracy on the given testing data

        _class = ["Hot dog", "Not hot dog"]
        j = 0
        total = 0
        print("Here are a list of inaccurately classified results:")
        for x, y in DataLoader(dataset=testing_data, batch_size=1, shuffle=True):
            x, y = x.to(self._device), y.to(self._device)
            predicted = argmax(softmax(self._model(x.to(self._device)), dim=1), dim=1)
            if predicted != y:
                j += 1
                print(f"Actual: {_class[y.item()]}\t\tPredicted: {_class[predicted.item()]}")
                imshow(x[0])
            total += 1
        return 100 - 100 * j / total

    def dump_to(self, file_name: str) -> None:

        #This function dumps the trained model. 

        #file_name: The directory to save state files.

        save_path = join(getcwd(), file_name)
        save({"model_params": self._model.state_dict(),
              "optimiser_stats": self._optimiser.state_dict(),
              "acc": self.validation_acc,
              "loss": self.training_loss
              }, save_path)
        
    def load_from(self, path: str) -> None:

        #This function loads the dumped file back to the training framework

       #path: The path to the dumped file.
       
        state_dict = load(path, map_location=self._device)
        self._model.load_state_dict(state_dict["model_params"])
        self._optimiser.load_state_dict(state_dict["optimiser_stats"])
        self.validation_acc = state_dict["acc"]
        self.training_loss = state_dict["loss"]