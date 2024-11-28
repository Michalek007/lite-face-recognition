from models.lite_face import LiteFace
from models.shuffle_face_net import ShuffleFaceNet


class Model:
    @staticmethod
    def get(name: str, input_channels: int, image_size: tuple, **kwargs):
        if name == 'lite-face':
            return LiteFace(input_channels, image_size, **kwargs)
        else:
            return ShuffleFaceNet(input_channels, image_size, **kwargs)
