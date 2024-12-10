import torch
from torch import nn
from model import Model
from trainer import Trainer


if __name__ == '__main__':
    config = {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'f1': 512, 'f2': 128, 'm': 0.9, 'lr': 0.1, 'b': 16}
    IN_CHANNELS = 3
    IMAGE_H = 100
    IMAGE_W = 100
    LEARNING_RATE = config.get('lr')
    EPOCHS = 20
    BATCH_SIZE = config.get('b')
    MARGIN = config.get('m')
    # MODEL_NAME = 'lite_face'
    MODEL_NAME = 'lite_face_100'
    DATA_DIR = '../data_mtcnn' if MODEL_NAME == 'lite_face_100' else None

    model = Model.get(MODEL_NAME, IN_CHANNELS, (IMAGE_H, IMAGE_W), c1=config.get('c1'), c2=config.get('c2'), c3=config.get('c3'), c4=config.get('c4'), f1=config.get('f1'), f2=config.get('f2'))
    loss_fn = nn.CosineEmbeddingLoss(margin=MARGIN, reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print(model)
    print(optimizer)

    trainer = Trainer(model, optimizer, scheduler, loss_fn, EPOCHS, BATCH_SIZE, LEARNING_RATE, MODEL_NAME, margin=MARGIN, data_dir=DATA_DIR)

    train = False
    save_onnx = False
    load_from_checkpoint = False
    save_to_checkpoint = True
    save_results = True
    if train:
        accuracy, average_train_loss, average_test_loss = trainer.run(load_from_checkpoint, save_to_checkpoint, save_results)
        trainer.save_model()

    else:
        trainer.load_model(best_model=False)
        model.eval()
        accuracy = trainer.test_loop()
        print(f'Accuracy: {(100 * accuracy):>0.1f}%')

    if save_onnx:
        trainer.save_model_to_onnx()