from models.lite_face import LiteFace
from models.lite_face_100 import LiteFace100


class Model:
    @staticmethod
    def get(name: str, input_channels: int, image_size: tuple, **kwargs):
        if name == 'lite_face':
            return LiteFace(input_channels, image_size, **kwargs)
        else:
            return LiteFace100(input_channels, image_size, **kwargs)
