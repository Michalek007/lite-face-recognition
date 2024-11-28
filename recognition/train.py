import torch
from torch import nn
from model import Model
from trainer import Trainer


if __name__ == '__main__':
    IN_CHANNELS = 3
    IMAGE_H = 100
    IMAGE_W = 100
    LEARNING_RATE = 0.01
    EPOCHS = 5
    BATCH_SIZE = 16
    MARGIN = 0.5
    # MODEL_NAME = 'lite_face'
    MODEL_NAME = 'lite_face_100'
    DATA_DIR = '../data_mtcnn' if MODEL_NAME == 'lite_face_100' else None

    model = Model.get(MODEL_NAME, IN_CHANNELS, (IMAGE_H, IMAGE_W))
    loss_fn = nn.CosineEmbeddingLoss(margin=MARGIN)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    print(model)
    print(optimizer)

    trainer = Trainer(model, optimizer, loss_fn, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, margin=MARGIN, data_dir=DATA_DIR)

    train = True
    save_onnx = False
    load_from_checkpoint = True
    save_to_checkpoint = True
    save_results = True
    if train:
        accuracy, average_train_loss, average_test_loss = trainer.run(load_from_checkpoint, save_to_checkpoint, save_results)
        trainer.save_model()

    else:
        trainer.load_model(best_model=True)
        model.eval()
        accuracy = trainer.test_loop()
        print(f'Accuracy: {(100 * accuracy):>0.1f}%')

    if save_onnx:
        trainer.save_model_to_onnx()
