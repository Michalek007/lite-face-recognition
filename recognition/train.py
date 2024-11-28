import torch
from torch import nn
from model import Model
from trainer import Trainer


if __name__ == '__main__':
    IN_CHANNELS = 3
    IMAGE_H = 250
    IMAGE_W = 250
    LEARNING_RATE = 0.01
    EPOCHS = 5
    BATCH_SIZE = 16
    model_name = 'lite-face'

    model = Model.get(model_name, IN_CHANNELS, (IMAGE_H, IMAGE_W))
    loss_fn = nn.CosineEmbeddingLoss(margin=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    print(model)
    print(optimizer)

    trainer = Trainer(model, optimizer, loss_fn, EPOCHS, BATCH_SIZE, LEARNING_RATE, model_name)

    train = True
    save_onnx = False
    load_from_checkpoint = False
    save_to_checkpoint = True
    save_results = True
    if train:
        accuracy, average_train_loss, average_test_loss = trainer.run(load_from_checkpoint, save_to_checkpoint, save_results)
        trainer.save_model()

    if not train:
        trainer.load_model(best_model=True)
        model.eval()
        accuracy = trainer.test_loop()
        print(f'Accuracy: {(100 * accuracy):>0.1f}%')

    if save_onnx:
        torch_input = torch.randn(1, IN_CHANNELS, IMAGE_H, IMAGE_W)
        torch.onnx.export(model, torch_input, f'{model_name}.onnx')
