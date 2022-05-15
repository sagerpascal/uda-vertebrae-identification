import numpy as np
import torch

from .classifiers import DRNSegPixelClassifier3D, DRNSegPixelClassifier2D
from .encoders import UnetDetection, UnetIdentification


def get_models(mode, input_ch, n_class, is_data_parallel=False):
    def get_model_list():
        if mode == "detection":
            encoder = UnetDetection(in_channels=input_ch, out_channels=16)
            head = DRNSegPixelClassifier3D(in_channels=16, out_channels=n_class)
        elif mode == "identification":
            encoder = UnetIdentification(in_channels=input_ch, filters=16)
            head = DRNSegPixelClassifier2D(in_channels=16, out_channels=n_class)
        else:
            raise NotImplementedError("Only detection w. U-Net / identification w. U-Net supported!")

        return encoder, head

    model_list = get_model_list()

    if is_data_parallel:
        return [torch.nn.DataParallel(x) for x in model_list]
    else:
        return model_list


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_optimizer(model_parameters, opt, lr, momentum, weight_decay):
    if opt == "sgd":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model_parameters), lr=lr, momentum=momentum,
                               weight_decay=weight_decay)
    elif opt == "adadelta":
        return torch.optim.Adadelta(filter(lambda p: p.requires_grad, model_parameters), lr=lr,
                                    weight_decay=weight_decay)
    elif opt == "adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model_parameters), lr=lr)
    else:
        raise NotImplementedError("Only (Momentum) SGD, Adadelta, Adam are supported!")


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
