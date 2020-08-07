from gaptrain.configurations import ConfigurationSet


class Data(ConfigurationSet):

    def __init__(self, *args, name):
        super().__init__(*args, name=name)
