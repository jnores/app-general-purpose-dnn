#!/usr/bin/python3
import argparse
from configFile import *

from pytorchUtils import DNN_Model
import imageUtils

from http.server import HTTPServer
from RequestHandler import RequestHandler

from threading import Lock



class Server(HTTPServer):
    def __init__(self,server_address,requestHandlerClass,model,imageUtils):
        HTTPServer.__init__(self,server_address, requestHandlerClass)
        self.img_utils = imageUtils
        self.model = model
        self.work_lock = Lock()

def run(ip, port):
    print('starting server...(',ip,':',str(port),')')
     
    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
     
    server_address = (ip, port)
    model = DNN_Model(THRESHOLD)

    httpd = Server(server_address, RequestHandler,model,imageUtils)
    print('running single request server...')
    httpd.serve_forever()

# MAIN
# Se configuran los parametros disponibles y se procesa la entrada.
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--port','-p', type=int, default=DEFAULT_PORT,
                   help='numero de puerto (default: '+str(DEFAULT_PORT)+')')
parser.add_argument('--ip', default=DEFAULT_HOST,
                   help='ip de trabajo (default: '+str(DEFAULT_HOST)+')')
args = parser.parse_args()
# print(args)
#Inicia el server con los parametros pasados o los valores por default.
run(args.ip, args.port)