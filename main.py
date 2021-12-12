import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['QT_DRIVER'] = 'pyside2'

import tensorflow

tensorflow.get_logger().setLevel('ERROR')

from src.gui.app import App


def main():
    sys.exit(App().run())


if __name__ == '__main__':
    main()
