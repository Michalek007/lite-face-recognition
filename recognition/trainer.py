import torch
from torch import nn
import shutil
import os
from pathlib import Path
import logging
from dataset import Dataset


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, loss_fn, epochs: int, batch_size: int, learning_rate: float, model_name: str, **config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cp_dir_number = None
        self.config = config

        self.train_data, self.val_data, self.test_data = Dataset.load_datasets(transform_to_tensors=True, data_dir=config.get('data_dir'))
        self.train_dataloader, self.val_dataloader, self.test_dataloader = Dataset.get_dataloaders(self.train_data, self.val_data, self.test_data, batch_size=self.batch_size)

        self.file = f'{model_name}.pt'
        self.best_model_file = f'best_model_{self.file}'
        self.cp_file = f'cp_{self.file}'
        self.cp_optimizer_file = f'cp_optimizer_{self.file}'
        self.results_file = f'{model_name}_results.txt'
        self.logs_file = f'{model_name}_epochs_results.log'
        self.onnx_file = f'{model_name}.onnx'
        self.cp_scheduler_file = f'cp_scheduler_{self.file}'

        files = (self.file, self.best_model_file, self.cp_file, self.cp_optimizer_file, self.results_file, self.logs_file, self.onnx_file, self.cp_scheduler_file)

        self.results_dir = Path(f'results/{model_name}/')
        if not self.results_dir.exists():
            self.results_dir.mkdir()
        self.file, self.best_model_file, self.cp_file, self.cp_optimizer_file, self.results_file, self.logs_file, self.onnx_file, self.cp_scheduler_file\
            = map(lambda arg: str(Path.joinpath(self.results_dir, arg)), files)

        self.logging_enable = True
        if self.logging_enable:
            logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', filename=self.logs_file, level=logging.INFO)

        assert self.config.get('margin') is not None
        assert self.config.get('diff_loss_weight') is not None

    def run(self, load_from_checkpoint: bool = False, save_to_checkpoint: bool = True, save_results: bool = True):

        if load_from_checkpoint:
            self.load_checkpoint()

        best_loss = 0
        accuracy, average_train_loss, average_test_loss = 0.0, 0.0, 0.0
        if self.logging_enable:
            logging.info(f"Training of model: f'lr:{self.learning_rate};batch_size:{self.batch_size};epochs:{self.epochs}; {self.model.name}'")
        for t in range(self.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")

            average_train_loss = self.train_loop()
            accuracy, average_test_loss = self.validation_loop()

            # Learning rate adjustment
            if self.scheduler:
                self.scheduler.step()

            if self.logging_enable:
                logging.info(f"Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f} after {t+1} epoch. ")

            if save_to_checkpoint:
                if average_test_loss < best_loss:
                    torch.save(self.model.state_dict(), self.best_model_file)
                    best_loss = average_test_loss
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
            img1_pred, img2_pred = self.model(img1), self.model(img2)
            target[target == 0] = -1
            loss = self.loss_fn(img1_pred, img2_pred, target)
            loss[target==-1] *= self.config.get('diff_loss_weight')
            loss = loss.sum()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * self.batch_size + len(img1)
                print(f"loss: {loss:>8f}  [{current:>5d}/{dataset_len:>5d}]")

        running_loss /= dataset_len
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
                print(distance)
                print(target)
                xor = torch.logical_xor(distance < 0.9, target)
                correct += xor.sum().item()

                target[target == 0] = -1
                # test_loss += self.loss_fn(img1_pred, img2_pred, target).item()
                loss = self.loss_fn(img1_pred, img2_pred, target)
                loss[target == -1] *= self.config.get('diff_loss_weight')
                test_loss += loss.sum().item()

                print(f"accuracy: {(100*correct/((batch+1)*self.batch_size)):>0.1f}% [{batch * self.batch_size + len(img1):>5d}/{dataset_len:>5d}]")

        test_loss /= dataset_len
        correct /= dataset_len
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss

    def test_loop(self, **kwargs):
        self.model.eval()
        dataset_len = len(self.test_dataloader.dataset)
        batches_count = len(self.test_dataloader)
        correct = 0
        cosine_similarity = nn.CosineSimilarity()

        margin = kwargs.get("margin")
        if not margin:
            margin = self.config.get("margin")
        with torch.no_grad():
            for batch, (img1, img2, target) in enumerate(self.test_dataloader):
                img1_pred, img2_pred = self.model(img1), self.model(img2)
                distance = cosine_similarity(img1_pred, img2_pred)
                print(distance)
                print(target)
                xor = torch.logical_xor(distance < margin, target)
                correct += xor.sum().item()
                print(f"accuracy: {(100*correct/((batch+1)*self.batch_size)):>0.1f}% [{batch * self.batch_size + len(img1):>5d}/{dataset_len:>5d}]")

        correct /= dataset_len
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")
        return correct

    # def metric(self, img1_pred, img2_pred, target):
    #     cosine_similarity = nn.CosineSimilarity()
    #     distance = cosine_similarity(img1_pred, img2_pred)
    #     xor = torch.logical_xor(distance < self.config['margin'], target)
    #     correct = xor.sum().item()
    #     return correct
    #
    # def loss(self, img1_pred, img2_pred, target):
    #     target[target == 0] = -1
    #     loss = self.loss_fn(img1_pred, img2_pred, target).item()
    #     return loss

    def save_checkpoint(self):
        torch.save(self.model.state_dict(), self.cp_file)
        torch.save(self.optimizer.state_dict(), self.cp_optimizer_file)
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), self.cp_scheduler_file)
        path = self.results_dir.joinpath('models')
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
        shutil.copy(self.cp_file, path)
        shutil.copy(self.cp_optimizer_file, path)
        shutil.copy(self.best_model_file, path)
        if self.scheduler:
            shutil.copy(self.cp_scheduler_file, path)

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(self.cp_file, weights_only=True))
        self.optimizer.load_state_dict(torch.load(self.cp_optimizer_file))
        if self.scheduler:
            self.scheduler.load_state_dict(torch.load(self.cp_scheduler_file))

    def load_model(self, best_model: bool = True):
        model_file = self.best_model_file if best_model else self.file
        self.model.load_state_dict(torch.load(model_file, weights_only=True))

    def save_model(self):
        torch.save(self.model.state_dict(), self.file)

    def save_model_to_onnx(self):
        torch_input = torch.randn(1, self.model.input_channels, self.model.input_height, self.model.input_width)
        torch.onnx.export(self.model, torch_input, self.onnx_file)

    def save_results(self, accuracy: float, average_train_loss: float, average_test_loss: float):
        with open(self.results_file, 'a') as f:
            f.write(f'lr:{self.learning_rate};batch_size:{self.batch_size};epochs:{self.epochs}; {self.model.name};dir_number:{self.cp_dir_number}\n')
            f.write(str(self.model))
            f.write(f' Accuracy: {(100 * accuracy):>0.1f}%; Average train loss: {average_train_loss:>0.8f}; Average test loss: {average_test_loss:>0.8f}')
            f.write('\n\n')
