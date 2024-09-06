import argparse

from model import ModelClassification
from logger import logger


def arg_parser():
    parser = argparse.ArgumentParser(prog='check disk module',
                                     description="Утилита для предсказания по дискам,\
                                      установленным в серверах, для выдачи прогноза их выхода из строя.")
    subparsers = parser.add_subparsers()

    def train_model():
        model = ModelClassification()
        model.save_trained_model()
        print("Model've trained")

    def check_disk_health():
        params = arguments.param_a, arguments.param_b, arguments.param_c, arguments.param_d
        model = ModelClassification()
        pred = model.get_prediction(params)
        print('Disk is good') if pred == 1 else print('Disk is not healthy')
        print("Disk health done")

    train_service = subparsers.add_parser('train_model')
    train_service.set_defaults(func=train_model)
    check_service = subparsers.add_parser('check_disk')
    check_service.set_defaults(func=check_disk_health)

    # arguments for train_model service
    ### path to data

    # arguments for check_disc_service
    check_service.add_argument('-a', '--param_a',
                               help='Disk parameters',
                               required=True,
                               type=int)
    check_service.add_argument('-b', '--param_b',
                               help='Disk parameters',
                               required=True,
                               type=int)
    check_service.add_argument('-c', '--param_c',
                               help='Disk parameters',
                               required=True,
                               type=int)
    check_service.add_argument('-d', '--param_d',
                               help='Disk parameters',
                               required=True,
                               type=int)
    try:
        arguments = parser.parse_args()
        arguments.func()
    except SystemExit as e:
        log_msg = 'Check quantity of arguments, print "--help"'
        print(log_msg)
        logger.error(f'{e}: {log_msg}')