from typing import Union
import math


def params_to_tuple(kernel_size: Union[int, tuple], stride: Union[int, tuple], padding: Union[int, tuple]):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    return kernel_size, stride, padding


def get_output_size(input_size: tuple, kernel_size: tuple, stride: tuple, padding: tuple):
    output_height = (input_size[0] - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
    output_width = (input_size[1] - kernel_size[1] + 2 * padding[1]) / stride[1] + 1
    return output_height, output_width


def get_conv_output_size(out_channels: int, input_size: tuple, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0):
    """
    Returns:
        flatten output len, (output_height, output_width)
    """
    kernel_size, stride, padding = params_to_tuple(kernel_size, stride, padding)

    output_height, output_width = get_output_size(input_size, kernel_size, stride, padding)
    output_height, output_width = int(output_height), int(output_width)
    return int(out_channels * output_height * output_width), (output_height, output_width)


def get_max_pool_output_size(in_channels: int, input_size: tuple, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None, padding: Union[int, tuple] = 0, ceil_mode: bool = False):
    """
    Returns:
        flatten output len, (output_height, output_width)
    """
    kernel_size, stride, padding = params_to_tuple(kernel_size, stride, padding)

    if not stride:
        stride = kernel_size
    output_height, output_width = get_output_size(input_size, kernel_size, stride, padding)
    output_height, output_width = (math.ceil(output_height), math.ceil(output_width)) if ceil_mode else (int(output_height), int(output_width))
    return int(in_channels * output_height * output_width), (output_height, output_width)


def get_model_params_count(model):
    """
    Returns:
        model params count
    """
    total_params = sum(p.numel() for p in model.parameters())
    return total_params
