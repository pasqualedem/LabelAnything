import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, required=True, default='Exp')
parser.add_argument('--file', type=str, help='Set the config file')
parser.add_argument('--dir', default=None, help='Set the local tracking directory')


