from datetime import datetime
import os

class Logger():
    def __init__(self, name=None, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "logs")
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d-%H%M%S")

        if name == None:
            name = self.dt_string
        else:
            name = '{}_{}'.format(name, self.dt_string)
        self.log_file = '{}/text/{}.txt'.format(path, name)
        self.log_file_aux = '{}/text/{}_aux.txt'.format(path, name)
        print("logs are written to {}".format(self.log_file))

    def write_text(self, txt, silent=False):
        with open(self.log_file, 'a') as f:
            f.write(txt)
            f.write("\n")

        if not silent:
            print(txt)

    def write_text_aux(self, txt, silent=True):
        with open(self.log_file_aux, 'a') as f:
            f.write(txt)
            f.write("\n")

        if not silent:
            print(txt)

if __name__ == '__main__':
    logger = Logger()
    logger.write_text("1")
    logger.write_text("2")