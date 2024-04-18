import importlib as _importlib

submodules = ['optimize']

__all__ = submodules


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'myscipy.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'scipy' has no attribute '{name}'"
            )
