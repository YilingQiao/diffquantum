from datetime import datetime

class Logger():
    def __init__(self, path="./logs/"):
        self.path = path
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d_%H%M%S")

        self.log_file = '{}text/{}.txt'.format(path, self.dt_string)
        self.log_file_aux = '{}text/{}_aux.txt'.format(path, self.dt_string)
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