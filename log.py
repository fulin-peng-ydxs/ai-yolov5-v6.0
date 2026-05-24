# coding: utf-8

import logging
from logging import handlers
import os

basedir = os.path.abspath(os.path.dirname(__file__))
log_path = basedir + os.sep + "xiaohudao_test.log"

logger = logging.getLogger(log_path)
fmt = logging.Formatter('%(asctime)s,%(process)d,%(name)s,%(levelname)s,%(filename)s:%(lineno)d,%(message)s')
sh = logging.StreamHandler()
sh.setFormatter(fmt)

th = handlers.RotatingFileHandler(filename=log_path, maxBytes=30*1024*1024, backupCount=5, encoding="utf-8")
th.setFormatter(fmt)
logger.addHandler(sh)
logger.addHandler(th)

logger.setLevel(logging.INFO)

logger.info('project start log')

