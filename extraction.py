from torch.nn import Module
import torch
import os
from os import makedirs
from os.path import exists, join
from torch.nn import Conv2d, Linear, LSTM
from torch.utils.data import DataLoader
from typing import Union, Dict, List
from shutil import rmtree
from itertools import product
import numpy as np
from time import time
import pickle
from models.volo import ClassBlock, Transformer, Outlooker

if os.name == 'nt':  # running on windows:
    import win32file
    win32file._setmaxstdio(2048)


class LatentRepresentationCollector:
    """This Object collects the latent representation from all layers.
    """

    def __init__(self, model: Module,
                 savepath: str,
                 save_instantly: bool = True,
                 downsampling: int = None,
                 save_per_position: bool = False,
                 overwrite: bool = False):
        """

        Args:
            model:              this is a pyTorch-Module
            savepath:           the filepath points to a folder where latent representations will be stored.
                                For storage a subfolder will be created.
            save_instantly:     if true, the data will be saved incrementally with a save checkpoint at each batch.
            downsampling:       downsample the latent representation to a height and width equal to the downsampling value.
            save_per_position:  saves a dataset per layer per position of the feature map instead of saving the feature maps downsamples as a whole.
        """
        self.savepath = savepath
        self.downsampling = downsampling
        self.save_per_position = save_per_position
        self.overwrite = overwrite
        self.pre_exists = False
        if exists(savepath) and overwrite:
            print('Found previous extraction in folder, removing it...')
            rmtree(savepath)
        if not exists(savepath):
            makedirs(self.savepath)
        else:
            self.pre_exists = True
        self.layers = self.get_layers_recursive(model)
        for name, layer in self.layers.items():
            if isinstance(layer, Conv2d) \
                    or isinstance(layer, Linear) \
                    or isinstance(layer, LSTM) \
                    or isinstance(layer, Outlooker) \
                    or isinstance(layer, Transformer) \
                    or isinstance(layer, ClassBlock) \
                    or "adder" in type(layer).__name__.lower():
                self._register_hooks(layer=layer,
                                     layer_name=name,
                                     interval=1)
        self.save_instantly = save_instantly

        self.logs = {
            'train': {},
            'eval': {}
        }
        self.record = True

    def _record_stat(self, activations_batch: torch.Tensor, layer: Module, training_state: str):
        """This function is called in the forward-hook to all convolutional and linear layers.

        Args:
            activations_batch:  the batch of data
            layer:              the module object, the latent representations are recorded.
            training_state:     state of training, may be either "eval" or "train"

        Returns:
            Returns nothing, this hook is side-effect free
        """
        if activations_batch.dim() == 4:  # conv layer (B x C x H x W)
            if self.downsampling is not None:
                #activations_batch = torch.nn.functional.interpolate(activations_batch, self.downsampling, mode="nearest")
                activations_batch = torch.nn.functional.adaptive_avg_pool2d(activations_batch, (self.downsampling, self.downsampling))
            if not self.save_per_position:
                activations_batch = activations_batch.view(activations_batch.size(0), -1)
        batch = activations_batch.cpu().numpy()
        if not self.save_instantly:
            if layer.name not in self.logs[training_state]:
                self.logs[training_state][layer.name] = batch
            else:
                self.logs[training_state][layer.name] = np.vstack((self.logs[training_state][layer.name], batch))
        elif self.save_per_position and len(batch.shape) == 4:
            for (i, j) in product(range(batch.shape[2]), range(batch.shape[3])):
                position = batch[:, :, i, j]
                saveable = position.squeeze()
                savepath = self.savepath + '/' + training_state + '-' + layer.name + f'-({i},{j})' + '.p'
                if not exists(savepath):
                    if layer.name not in self.logs[training_state]:
                        self.logs[training_state][layer.name] = {}
                    self.logs[training_state][layer.name][(i, j)] = savepath

                with open(self.logs[training_state][layer.name][(i, j)], 'ab') as fp:
                    pickle.dump(saveable, file=fp)

        else:
            savepath = self.savepath+'/'+training_state+'-'+layer.name+'.p'
            if not exists(savepath):
                self.logs[training_state][layer.name] = open(savepath, 'wb')
            pickle.dump(batch, file=self.logs[training_state][layer.name])

    def _register_hooks(self, layer: Module, layer_name: str, interval: int) -> None:
        """Register a forward hook on a given layer.

        Args:
            layer:          the module.
            layer_name:     name of the layer.
            interval:       unused variable, needed for compatibility.
        """
        layer.name = layer_name

        def record_layer_history(layer: torch.nn.Module, input, output):
            """Hook to register in `layer` module."""

            if not self.record:
                return

            # Increment step counter
            # layer.forward_iter += 1

            training_state = 'train' if layer.training else 'eval'
            activations_batch = output.data
            self._record_stat(activations_batch, layer, training_state)

        layer.register_forward_hook(record_layer_history)

    def get_layer_from_submodule(self, submodule: torch.nn.Module,
                                 layers: dict, name_prefix: str = '') -> Dict[str, Module]:
        """Finds all linear and convolutional layers in a network structure.

        The algorithm is recursive.

        Args:
            submodule:      the current submodule.
            layers:         the dictionary containing all layers found so far.
            name_prefix:    the prefix of the layers name. The prefix resembled the position in
                            the networks structure.

        Returns:
            the layers stored in a dictionary.
        """
        if len(submodule._modules) > 0  and not \
                   isinstance(submodule, Transformer) and not \
                   isinstance(submodule, Outlooker) and not \
                   isinstance(submodule, ClassBlock):
            for idx, (name, subsubmodule) in \
                              enumerate(submodule._modules.items()):
                new_prefix = name if name_prefix == '' else name_prefix + \
                                                            '-' + name
                self.get_layer_from_submodule(subsubmodule, layers, new_prefix)
            return layers
        else:
            layer_name = name_prefix
            layer_type = layer_name
            if not isinstance(submodule, Conv2d) and not \
                   isinstance(submodule, Transformer) and not \
                   isinstance(submodule, Outlooker) and not \
                   isinstance(submodule, ClassBlock) and not \
                   isinstance(submodule, Linear) and not \
                   isinstance(submodule, LSTM) and not "adder" in type(submodule).__name__.lower():
                print(f"Skipping {layer_type}")
                return layers
            layers[layer_name] = submodule
            print('added layer {}'.format(layer_name))
            return layers

    def get_layers_recursive(self, modules: Union[List[torch.nn.Module], torch.nn.Module]) -> Dict[str, Module]:
        """Recursive search algorithm for finding convolutional an linear layers

        Args:
            modules: maybe a single (sub)-module or a List of modules

        Returns:
            a dictionary that maps layer names to modules
        """
        layers = {}
        if not isinstance(modules, list) and not hasattr(
                modules, 'out_features'):
            # is a model with layers
            # check if submodule
            submodules = modules._modules  # OrderedDict
            layers = self.get_layer_from_submodule(modules, layers, '')
        else:
            for module in modules:
                layers = self.get_layer_from_submodule(module, layers, '')
        return layers

    def save(self, model_log_path) -> None:
        """Saving the models latent representations.

        Args:
            model_log_path:     the path that logs the model.

        """
        with open(join(self.savepath, "model_pointer.txt"), "w+") as fp:
            fp.write(model_log_path)
        if not exists(self.savepath):
            makedirs(self.savepath)
        for mode, logs in self.logs.items():
            for layer_name, data in self.logs[mode].items():
                if isinstance(data, str):
                    continue
                if isinstance(data, np.ndarray):
                    with open(self.savepath+'/'+mode+'-'+layer_name+'.p', 'wb') as p:
                        pickle.dump(data, p)
                else:
                    if isinstance(data, dict):
                        for _, fp in data.items():
                            if hasattr(fp, "close"):
                                fp.close()
                    else:
                        if hasattr(fp, "close"):
                            data.close()


def extract_from_dataset(logger: LatentRepresentationCollector,
                         train: bool, model: Module, dataset: DataLoader, device: str) -> None:
    """Extract latent representations from a given classification dataset.

    Args:
        logger:     The logger that collects the latent representations.
        train:      Marks the subset as training or evalutation dataset
        model:      The model from which the latent representations need to be collected.
        dataset:    The dataset, may be a torch data-loader
        device:     The device the model is deployed on, maybe any torch compatible key.
    """
    if logger.pre_exists:
        print("Found existing latent representations and overwrite is disabled")
        return
    mode = 'train' if train else 'eval'
    correct, total = 0, 0
    old_time = time()
    with torch.no_grad():
        for batch, data in enumerate(dataset):
            if batch % 1 == 0 and batch != 0:
                print(batch, 'of', len(dataset), 'processing time', time() - old_time,' acc:', correct / total)
                old_time = time()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if len(labels.shape) > 1:
                _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()
            if 'labels' not in logger.logs[mode]:
                logger.logs[mode]['labels'] = labels.cpu().numpy()
            else:
                logger.logs[mode]['labels'] = np.hstack((logger.logs[mode]['labels'], labels.cpu().numpy()))
    print('accuracy:', correct/total)
