from functools import partial
import tempfile
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
import os
from pathlib import Path
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle

from trainer import Trainer
from dataset import Dataset
from model import Model


IN_CHANNELS = 3
IMAGE_H = 160
IMAGE_W = 160
MODEL_NAME = 'lite_face_100'


def load_data(data_dir="../data_mtcnn"):
    return Dataset.load_datasets(0.2, data_dir=data_dir)


def train_lwf(config, data_dir=None):
    net = Model.get(MODEL_NAME, IN_CHANNELS, (IMAGE_H, IMAGE_W), c2=config["c2"], c3=config["c3"], f2=config["f2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CosineEmbeddingLoss(margin=config["m"], size_average=False, reduction="sum")
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_subset, val_subset, testset = load_data(data_dir)
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["b"]), shuffle=True, num_workers=6
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["b"]), shuffle=True, num_workers=6
    )

    cosine_similarity = nn.CosineSimilarity()
    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, (img1, img2, target) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            img1_pred = net(img1)
            img2_pred = net(img2)
            target[target == 0] = -1
            loss = criterion(img1_pred, img2_pred, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / len(trainloader.dataset))
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, (img1, img2, target) in enumerate(valloader, 0):
            total += target.size(0)

            with torch.no_grad():
                img1_pred, img2_pred = net(img1), net(img2)
                distance = cosine_similarity(img1_pred, img2_pred)
                xor = torch.logical_xor(distance < config['m'], target)
                correct += xor.sum().item()

                target[target == 0] = -1
                loss = criterion(img1_pred, img2_pred, target)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "wb") as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss / len(valloader.dataset), "accuracy": correct / total},
                checkpoint=checkpoint,
            )

    print("Finished Training")


def test_accuracy(net, margin, device="cpu"):
    trainset, valset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=2
    )

    net.eval()
    dataset_len = len(testset)
    batches_count = len(testloader)
    correct = 0
    cosine_similarity = nn.CosineSimilarity()

    with torch.no_grad():
        for batch, (img1, img2, target) in enumerate(testloader):
            img1_pred, img2_pred = net(img1), net(img2)
            distance = cosine_similarity(img1_pred, img2_pred)
            xor = torch.logical_xor(distance < margin, target)
            correct += xor.sum().item()

    correct /= dataset_len
    return correct


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("../data_mtcnn")
    load_data(data_dir)
    config = {
        'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'f1': 512, 'f2': 128,
        # "c2": tune.choice([2 ** i for i in range(2, 7)]),
        # "c3": tune.choice([2 ** i for i in range(2, 7)]),
        # "f2": tune.choice([2**i for i in range(2, 7)]),
        # "m": tune.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        'm': 0.8,
        "lr": tune.loguniform(1e-3, 1e-1),
        # "b": tune.choice([4, 8, 16]),
        "b": 16
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_lwf, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Model.get(MODEL_NAME, IN_CHANNELS, (IMAGE_H, IMAGE_W),
                                   c1=best_trial.config["c1"], c2=best_trial.config["c2"],
                                   c3=best_trial.config["c3"], c4=best_trial.config["c4"],
                                   f1=best_trial.config["f1"], f2=best_trial.config["f2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="accuracy", mode="max")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, best_trial.config["m"], device)
        print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
