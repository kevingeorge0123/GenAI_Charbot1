import logging
import os

logfile= "loggingfile.log"
logpath= os.path.join(os.getcwd(),"logs")
os.makedirs(logpath,exist_ok=True)
logfilepath = os.path.join(logpath,logfile)

logging.basicConfig(level=logging.INFO, 
                    filename=logfilepath,
                    format='[%(asctime)s]:%(message)s:')