import torch
from torch import nn
import shutil
import os
from pathlib import Path
import logging
from dataset import Dataset


class Trainer:
    def __init__(self, model: nn.Module, optimizer, loss_fn, epochs: int, batch_size: int, learning_rate: float, filename: str, **config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.filename = f'{filename}.pt'
        self.cp_dir_number = None
        self.config = config

        self.train_data, self.val_data, self.test_data = Dataset.load_datasets(transform_to_tensors=True)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = Dataset.get_dataloaders(self.train_data, self.val_data, self.test_data, batch_size=self.batch_size)

        self.logging_enable = True
        if self.logging_enable:
            logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename='epochs_results.log',
                                level=logging.INFO)

        self.best_model_filename = f'best_model_{self.filename}'

    def run(self, load_from_checkpoint: bool = False, save_to_checkpoint: bool = True, save_results: bool = True):

        if load_from_checkpoint:
            self.load_checkpoint()

        last_accuracy = 0
        accuracy, average_train_loss, average_test_loss = 0.0, 0.0, 0.0
        if self.logging_enable:
            logging.info(f"Training of model: f'lr:{self.learning_rate};batch_size:{self.batch_size};epochs:{self.epochs}; {self.model.name}'")
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            average_train_loss = self.train_loop()
            accuracy, average_test_loss = self.validation_loop()

            if self.logging_enable:
                logging.info(f"Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f} after {t+1} epoch. ")

            if save_to_checkpoint:
                if accuracy > last_accuracy:
                    torch.save(self.model.state_dict(), f'best_model_{self.filename}')
                    last_accuracy = accuracy
                self.save_checkpoint()

        accuracy = self.test_loop()

        if save_results:
            self.save_results(accuracy, average_train_loss, average_test_loss)
        print("Done!")
        return accuracy, average_train_loss, average_test_loss

    def train_loop(self):
        self.model.train()
        dataset_len = len(self.train_dataloader.dataset)
        batches_count = len(self.train_dataloader)
        running_loss = 0

        for batch, (img1, img2, target) in enumerate(self.train_dataloader):
            # Compute prediction and loss
            img1_pred = self.model(img1)
            img2_pred = self.model(img2)
            target[target == 0] = -1
            loss = self.loss_fn(img1_pred, img2_pred, target)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(img1)
                print(f"loss: {loss:>8f}  [{current:>5d}/{dataset_len:>5d}]")

        running_loss /= batches_count
        print(f"Train loss: {running_loss:>8f}")
        return running_loss

    def validation_loop(self):
        self.model.eval()
        dataset_len = len(self.val_dataloader.dataset)
        batches_count = len(self.val_dataloader)
        test_loss, correct = 0, 0
        cosine_similarity = nn.CosineSimilarity()

        with torch.no_grad():
            for batch, (img1, img2, target) in enumerate(self.val_dataloader):
                img1_pred, img2_pred = self.model(img1), self.model(img2)
                distance = cosine_similarity(img1_pred, img2_pred)
                xor = torch.logical_xor(distance < 0.5, target)
                correct += xor.sum().item()

                target[target == 0] = -1
                test_loss += self.loss_fn(img1_pred, img2_pred, target).item()

                print(f"accuracy: {(100*correct/((batch+1)*self.batch_size)):>0.1f}% [{batch * self.batch_size + len(img1):>5d}/{dataset_len:>5d}]")

        test_loss /= batches_count
        correct /= dataset_len
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss

    def test_loop(self):
        self.model.eval()
        dataset_len = len(self.test_dataloader.dataset)
        batches_count = len(self.test_dataloader)
        correct = 0
        cosine_similarity = nn.CosineSimilarity()

        with torch.no_grad():
            for batch, (img1, img2, target) in enumerate(self.test_dataloader):
                img1_pred, img2_pred = self.model(img1), self.model(img2)
                distance = cosine_similarity(img1_pred, img2_pred)
                xor = torch.logical_xor(distance < 0.6, target)
                correct += xor.sum().item()
                print(f"accuracy: {(100*correct/((batch+1)*self.batch_size)):>0.1f}% [{batch * self.batch_size + len(img1):>5d}/{dataset_len:>5d}]")

        correct /= dataset_len
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
        return correct

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), f'cp_{self.filename}')
        torch.save(self.optimizer.state_dict(), f'cp_optimizer_{self.filename}')
        path = Path(f'models')
        if not path.exists():
            path.mkdir()
        dirs = os.listdir(path)
        if '0' not in dirs:
            path = path.joinpath('0')
            path.mkdir(exist_ok=True)
            self.cp_dir_number = '0'
        else:
            if self.cp_dir_number is None:
                dir_number = sorted(map(int, filter(lambda arg: arg.isnumeric(), dirs)))[-1] + 1
                self.cp_dir_number = str(dir_number)
            path = path.joinpath(self.cp_dir_number)
            path.mkdir(exist_ok=True)
        shutil.copy(f'cp_{self.filename}', path)
        shutil.copy(f'cp_optimizer_{self.filename}', path)
        shutil.copy(f'best_model_{self.filename}', path)

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(f'cp_{self.filename}', weights_only=True))
        self.optimizer.load_state_dict(torch.load(f'cp_optimizer_{self.filename}'))

    def load_model(self, best_model: bool = True):
        model_filename = self.best_model_filename if best_model else self.filename
        self.model.load_state_dict(torch.load(model_filename, weights_only=True))

    def save_model(self):
        torch.save(self.model.state_dict(), self.filename)

    def save_results(self, accuracy: float, average_train_loss: float, average_test_loss: float):
        with open('results.txt', 'a') as f:
            f.write(f'lr:{self.learning_rate};batch_size:{self.batch_size};epochs:{self.epochs}; {self.model.name};dir_number:{self.cp_dir_number}\n')
            f.write(str(self.model))
            f.write(f' Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f}')
            f.write('\n\n')
