import yaml
import argparse


class ConfigParser:

    def load_config_file(self, parser, config_file):
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)

        for param, value in config_dict.items():
            parser.add_argument(
                "--{}".format(param),
                type=type(value),
                default=value,
                help=argparse.SUPPRESS
            )

parameters = ConfigParser()
