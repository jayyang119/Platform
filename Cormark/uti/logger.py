import os
import sys
import pandas as pd
import traceback
from datetime import datetime

from Path import DATABASE_PATH, DAILY_PATH


class Logger:
    def __init__(self):
        self.daily_path = DAILY_PATH
        self.database_path = DATABASE_PATH
        self.loglist = []
        self.time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        self.date = self.time[:8]
        self.filepath = os.path.join(self.database_path, f'Log/{self.date}/{self.time}.txt')
        self.NOW_STR = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

    def info(self, message=''):
        self.create_folder(self.filepath)
        time = self.current_time()

        if type(message) != str:
            if type(message) == pd.DataFrame:
                print(time)
                print(message)
                with open(self.filepath, 'a+') as f:
                    f.write('\n' + time + '\n')
                    if type(message) != pd.DataFrame:
                        try:
                            f.write(str(message))
                        except Exception as e:
                            print(e)
                if type(message) == pd.DataFrame:
                    message.to_csv(self.filepath, sep=' ', index=True, header=True, mode='a')

                with open(self.filepath, 'a+') as f:
                    f.write('\n')
        else:
            print(time, message)
            with open(self.filepath, 'a+') as f:
                f.write(time + ' ' + message + '\n\n')

        self.loglist.append(time)
        self.loglist.append(message)

    def error(self, message=''):
        self.create_folder(self.filepath)
        time = self.current_time()
        error_type, error = sys.exc_info()[0:2]  # 打印错误类型，错误值
        position = traceback.extract_tb(sys.exc_info()[2])  # 出错位置

        print(time, 'Position:', position, '| Error type:', error_type, '| Error:', error)

        with open(self.filepath, 'a+') as f:
            f.write(time + ' ')
            f.write('Position: ')
            f.write(str(position))
            f.write(' | Error type: ')
            f.write(str(error_type))
            f.write(' | Error: ')
            f.write(str(error))
            f.write('\n\n')

        if type(message) != str:
            if type(message) == pd.DataFrame:
                print(time)
                print(message)
                with open(self.filepath, 'a+') as f:
                    f.write('\n' + time + '\n')
                message.to_csv(self.filepath, sep=' ', index=True, header=True, mode='a')
                with open(self.filepath, 'a+') as f:
                    f.write('\n')
        else:
            print(time, message)
            with open(self.filepath, 'a+') as f:
                f.write(time + ' ' + message + '\n')
                f.write('\n')

    @classmethod
    def current_time(cls):
        time = datetime.strftime(datetime.now(), '%H:%M:%S')
        return time

    @classmethod
    def create_folder(cls, path):
        parent_directory = os.path.dirname(path)
        if os.path.exists(path):
            pass
            # print(cls().current_time(), 'Logger directory created.')
        else:
            if os.path.exists(parent_directory):
                print(cls().current_time(), f'{path} created.')
            else:
                cls().create_folder(parent_directory)

            if '.' not in path:
                os.mkdir(path)
            else:
                open(path, mode='a').close()

if __name__ == '__main__':
    ll = Logger()
    print(ll.database_path)
