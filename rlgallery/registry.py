import yaml
import inspect
from rich.console import Console
from rich.table import Table


class Registry:
    """all the parameter are given in the __init__(), config.yaml can override the default value.
    all the class with decorator should be run while importing, then the build function will work. 
    """
    def __init__(self, name: str, config_path: str):
        self._name = name
        self._module_dict = {}
        self._config_path = config_path

    @property
    def module_dict(self):
        return self._module_dict

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self._module_dict.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    def _register_module(self, module):
        if inspect.isclass(module):
            module_name = module.__name__
        else:
            raise TypeError(
                f'A non-class object: {module}')
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if name in self._module_dict:
                existed_module = self._module_dict[name]
                raise KeyError(f'{name} is already registered in {self._name} '
                               f'at {existed_module.__module__}')
        self._module_dict[name] = module

    def register_module(self):
        def _register(module):
            self._register_module(module=module)
            return module
        return _register

    def build(self, cfg):
        assert cfg.get('type') is not None, "`type` must be in the cfg dict"
        args = cfg.copy()
        class_name = args.pop('type')
        module = self._module_dict.get(class_name)
        instance = module(**args) # we first initailize the class using __init__(), so in order to make simpler param transfer, all the attrs should have one default value in __init__()
        with open(self._config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        relevant_config = config_data.get(class_name, {})
        # Extend the class properties with the relevant config data
        for key, value in relevant_config.items():
            setattr(instance, key, value) # in python, instance attr can overide class attr, so here's instance not module
        return instance.create_true_instance()

MODELS = Registry("models", "config.yaml")
OPTIMIZERS = Registry("optimizers", "config.yaml")
RUNNERS = Registry("runners", "config.yaml")
ENVS = Registry("envs", "config.yaml")
ALGORITHMS = Registry("algorithms", "config.yaml")
LOGGRS = Registry("loggers", "config.yaml")
UTILS = Registry("utils", "config.yaml")

if __name__ == "__main__":
    test = Registry("test", "config.yaml")
    # @test.register_module()
    # class A:
    #     def __init__(self) -> None:
    #         self.Aa = self.nb

    #     def info(self, h):
    #         print('ww', h)
    #         print("self.Aa", self.Aa)
    #         print("self.yaml", self.yaml)
    
    a = test.build({'type': 'Runner'})

