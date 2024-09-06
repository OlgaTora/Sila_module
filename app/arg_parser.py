import argparse

from model import ModelClassification
from logger import logger


class ArgParser:
    def __init__(self):
        self.arguments = None
        self.parser = argparse.ArgumentParser(prog='check disk module',
                                              description="Утилита для предсказания по дискам,\
                                      установленным в серверах, для выдачи прогноза их выхода из строя.")
        self.subparsers = self.parser.add_subparsers(help='sub-services help')

    def train_model(self):
        params = self.arguments.path_to_file, self.arguments.delimeter
        model = ModelClassification()
        model.save_trained_model(params)
        print("Model've trained")

    def check_disk_health(self):
        params = (self.arguments.param_a,
                  self.arguments.param_b,
                  self.arguments.param_c,
                  self.arguments.param_d)
        model = ModelClassification()
        pred = model.get_prediction(params)
        print('Disk is good') if pred == 1 else print('Disk is not healthy')
        print("Disk health done")

    def train_model_service(self):
        train_service = self.subparsers.add_parser('train_model', help='Обучение модели',
                                                   description='Обучение модели на данных из указанного файла или директории')
        train_service.set_defaults(func=self.train_model)
        # arguments for train_model service
        train_service.add_argument('-p', '--path_to_file',
                                   help='Путь к файлу или директории с данными для обучения модели',
                                   required=True,
                                   type=str)
        train_service.add_argument('-d', '--delimeter',
                                   help='Разделитель для csv - файла. По умолчанию ;',
                                   default=';'
                                   )

    def check_disk_service(self):
        check_service = self.subparsers.add_parser('check_disk', help='Проверка диска',
                                                   description='Проверка одного диска на вероятность выхода из строя')
        check_service.set_defaults(func=self.check_disk_health)
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

    def parse(self):
        self.train_model_service()
        self.check_disk_service()
        try:
            self.arguments = self.parser.parse_args()
            self.arguments.func()
        except SystemExit as e:
            log_msg = 'Check quantity of arguments, print "--help"'
            print(log_msg)
            logger.error(f'{e}: {log_msg}')
