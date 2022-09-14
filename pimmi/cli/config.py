import yaml


class ConfigParser:

    def load_config_file(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        return config_dict

    def set_config_as_attributes(self, config_dict):
        for key, value in config_dict.items():
            self.__setattr__(key, value)


parameters = ConfigParser()
