'''
Created on 2011-3-29

@author: mac
'''
import logging
from logging.handlers import TimedRotatingFileHandler


class LogUtils(object):
    def __init__(self,fileName):
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(name)-12s %(levelname)-6s %(message)s',datefmt='%y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(fileName)
        self.logger.setLevel(logging.DEBUG)
        fileTimeHandler = TimedRotatingFileHandler(filename=fileName,when='D',interval=1)#every 2 days to rotate
        fileTimeHandler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fileTimeHandler.setFormatter(formatter)
        fileTimeHandler.suffix = "%Y%m%d.bak"
        self.logger.addHandler(fileTimeHandler);
    def log(self,content,level="INFO"):
        if level == "INFO":
            self.logger.info(content)
        else:
            self.logger.error(content)
            
if __name__ == "__main__":
    logger = LogUtils("/tmp/log.txt")
    import time
    while 1:
        logger.log("abc")
        time.sleep(1)
