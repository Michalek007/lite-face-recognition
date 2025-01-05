import torch
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple

from utils.nn_utils import get_conv_output_size, get_max_pool_output_size


class CParser:
    def __init__(self):
        self.flatten = nn.Flatten(start_dim=0)

    def tensor_to_array(self, name: str, tensor: torch.Tensor, save_to_file: bool = False):
        tensor = self.flatten(tensor)
        if tensor.dtype == torch.float:
            data_type = 'float'
        elif tensor.dtype == torch.int:
            data_type = 'int'
        else:
            data_type = 'undefined'
        c_array = f'{data_type} {name}[] = {{'

        c_array += ', '.join(map(lambda t: str(round(t.item(), 5)), tensor))

        c_array += '};'

        if save_to_file:
            with open(f'{name}.txt', 'w') as f:
                f.write(c_array)
                f.write('\n')
        return c_array

    def tensor_to_const_array(self, name: str, tensor: torch.Tensor, save_to_file: bool = False):
        return f'const {self.tensor_to_array(name, tensor, save_to_file=save_to_file)}'

    def output_testing(self, expected_output: torch.Tensor):
        c_str = self.tensor_to_array('expectedOutput', expected_output)
        c_str += f"""
for (size_t i=0;i<{len(self.flatten(expected_output))};++i){{
    printf("Output [%d]: %f\\n", i, output[i]);
    assert(equalFloatDefault(output[i], expectedOutput[i]));
}}"""
        return c_str

    def model(self, model: torch.nn.Module, input_size: tuple, save_to_file: bool = False):
        model_c_str = ''
        i, j, k, l = 0, 0, 0, 0
        for key, value in model.state_dict().items():
            name = key.split('.')[-1]
            if name == 'weight':
                name += str(j)
                j += 1
            elif name == 'bias':
                name += str(k)
                k += 1
            else:
                name += str(l)
                l += 1
            model_c_str += self.tensor_to_const_array(name, value) + '\n'
            i += 1

        i, j, k, l = 0, 0, 0, 0

        def get_value(param: Union[int, tuple]):
            if isinstance(param, int):
                return param, param
            return param

        layer_str = ''
        output_channels = 3
        output_size = input_size
        output_len = output_channels * output_size[0] * output_size[1]
        for _, layer in enumerate(model.modules()):
            if i == 0:
                i += 1
                continue
            if i == 1:
                input_array_name = 'input'
            else:
                input_array_name = f'output{i-1}'
            name = layer.__class__.__name__
            input_size = output_size
            if name == 'Conv2d':
                kernel_h, kernel_w = get_value(layer.kernel_size)
                stride_h, stride_w = get_value(layer.stride)
                padding_h, padding_w = get_value(layer.padding)
                output_len, output_size = get_conv_output_size(layer.out_channels, input_size, layer.kernel_size, layer.stride, layer.padding)
                output_channels = layer.out_channels

                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_ConvLayer({layer.in_channels}, {input_size[0]}, {input_size[0]}, {output_channels}, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, {padding_h}, {padding_w}, {input_array_name}, weight{j}, bias{k}, output{i});\n'
                j += 1
                k += 1
                i += 1
            elif name == 'MaxPool2d' or name == 'AvgPool2d':
                kernel_h, kernel_w = get_value(layer.kernel_size)
                stride_h, stride_w = get_value(layer.stride)
                padding_h, padding_w = get_value(layer.padding)
                output_len, output_size = get_max_pool_output_size(output_channels, input_size, layer.kernel_size, layer.stride, layer.padding, layer.ceil_mode)
                layer_str = f'float output{i}[{output_len}];\n'
                if name == 'MaxPool2d':
                    layer_str += f'CNN_MaxPool({output_channels}, {input_size[0]}, {input_size[1]}, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, {padding_h}, {padding_w}, {int(layer.ceil_mode)}, {input_array_name}, output{i});\n'
                else:
                    layer_str += f'CNN_AveragePool({output_channels}, {input_size[0]}, {input_size[1]}, {kernel_h}, {kernel_w}, {stride_h}, {stride_w}, {padding_h}, {padding_w}, {int(layer.ceil_mode)}, {input_array_name}, output{i});\n'
                i += 1
            elif name == 'AdaptiveAvgPool2d':
                continue
            elif name == 'Linear':
                output_len = layer.out_features
                output_channels = layer.out_features
                output_size = (1, 1)
                layer_str = f'float output{i}[{output_len}];\n' + \
                            f'CNN_FcLayer({layer.in_features}, {output_len}, {input_array_name}, weight{j}, bias{k}, output{i});\n'
                j += 1
                k += 1
                i += 1
            elif name == 'ReLU':
                layer_str = f'CNN_ReLU({output_len}, {input_array_name});\n'
            elif name == 'PReLU':
                layer_str = f'CNN_PReLU({output_channels}, {input_size[0]}, {input_size[1]}, {input_array_name}, weight{j});\n'
                j += 1
            elif name == 'LeakyReLU':
                layer_str = f'CNN_LeakyReLU({output_channels}, {input_size[0]}, {input_size[1]}, {input_array_name}, {layer.negative_slope});\n'
            elif name == 'BatchNorm2d':
                layer_str = f'CNN_BatchNorm({output_channels}, {input_size[0]}, {input_size[1]}, {input_array_name}, weight{j}, bias{k}, mean{l}, variance{l+1});\n'
                j += 1
                j += 1
                l += 2
            elif name == 'Softmax':
                if input_size == (1, 1):
                    layer_str = f'CNN_Softmax({output_len}, {input_array_name});\n'
                else:
                    layer_str = f'CNN_Softmax2D({output_channels}, {input_size[0]}, {input_size[1]}, {layer.dim-1}, {input_array_name});\n'
            elif name == 'Flatten':
                layer_str = ''
            model_c_str += layer_str

        if save_to_file:
            with open(f'model_c.txt', 'w') as f:
                f.write(model_c_str)
                f.write('\n')
        return model_c_str
