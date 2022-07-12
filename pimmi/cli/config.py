import yaml


class ConfigParser:

    def load_config_file(self, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        return config_dict


parameters = ConfigParser()
