import importlib
from functools import partial

import omegaconf
import inspect


def instantiate_from_config(config, target_key="name", target_params_key="params", **kwargs):
    """
    Instantiate an object from a config dict.
    Args:
        config (dict, omegaconf.dictconfig.DictConfig): Config dict.
        target_key (str): Key of the target object.
        target_params_key (str): Key of the target object's parameters.
    Returns:
        object: Instantiated object.
    """
    if not isinstance(config, dict):
        config = dict(config)
    params = dict(config.get(target_params_key, dict()))
    merge_params = {**params, **kwargs}
    return get_obj_from_str(config[target_key])(**merge_params)


def recurse_instantiate_from_config(config, target_key="name", target_params_key="params", **kwargs):
    """
    Recursively instantiate an object from a config dict.
    Args:
        config (dict, omegaconf.dictconfig.DictConfig): Config dict.
        target_key (str): Key of the target object.
        target_params_key (str): Key of the target object's parameters.
    Returns:
        object: Instantiated object.
    """
    if not isinstance(config, dict):
        config = dict(config)
    params = dict(config.get(target_params_key, dict()))

    for k, v in params.items():
        if isinstance(v, dict) or isinstance(v, omegaconf.dictconfig.DictConfig):
            params[k] = recurse_instantiate_from_config(v, target_key, target_params_key)
    merge_params = {**params, **kwargs}
    return get_obj_from_str(config[target_key])(**merge_params)


def get_obj_from_str(string, reload=False):
    """
    Get an object from a string.
    Args:
        string (str): String of the object.
        reload (bool): Whether to reload the module.
    Returns:
        Class: Class of the object.
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class ClassInstance:
    """
    Create a class instance that can be called to create multiple instances of the target class.
    Use this class to create a class instead of a instance of a class, when you want to create in
    recurse_instantiate_from_config function.
    """

    def __new__(cls, target: str or type, **kwargs):
        if isinstance(target, str):
            target = get_obj_from_str(target)
        elif isinstance(target, type):
            pass
        else:
            raise TypeError(f'Invalid type for target_class: {type(target)}')
        return partial(target, **kwargs)
        # return target


def fill_args_from_dict(func, args_dict):
    args = inspect.getfullargspec(func).args
    args_dict = {k: v for k, v in args_dict.items() if k in args}
    return partial(func, **args_dict)