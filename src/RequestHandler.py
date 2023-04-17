from http.server import BaseHTTPRequestHandler, HTTPServer

from pytorchUtils import DNN_Model
import imageUtils

import time
from urllib.parse import urlparse,parse_qs,unquote
import json
from json import JSONEncoder
import numpy
import base64

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def createObject(x,y,z,ancho,alto,largo,nombre):
    return {
        'x': x/100,
        'y': y/100,
        'z': z/100,
        'ancho': ancho/100,
        'alto': alto/100,
        'largo': largo/100,
        'nombre': nombre
        }

def myrange(start, end, step):
    elems=[]
    aux=start
    while aux<end:
        elems.append(aux)
        aux+=step
    while aux>start:
        elems.append(aux)
        aux-=step
    return elems

positions=myrange(.5,4.5,.5)
i=0
lastTime=0
def getPosition():
    global i
    global lastTime
    if (time.time()-lastTime) > 2: # cambia la posicion cada 2 segundos
        lastTime=time.time()
        i+=1
        i=i % len(positions)
    return {'x': 1.5, 'y': 1.5, 'z': positions[i]}

def resetPosition():
    global i
    global lastTime
    i=0;
    lastTime=time.time()
    return getPosition()


class RequestHandler(BaseHTTPRequestHandler):

    # POST
    def do_POST(self):
        # Procesar request
        if self.server.work_lock.locked():
            self.send_response(503)
            self.wfile.write("SERVER BUSY", "utf8")
            return
        with self.server.work_lock:
            # time.sleep(5)
            header_cType = self.headers.get('content-type')
            # print ("<debug>");
            # print (header_cType);
            # print ("</debug>");
            if 'application/json' in header_cType:
                length = int(self.headers.get('content-length'))
                data = self.rfile.read(length)
            else:
                self.send_response(400)
                self.send_header('Content-type','application/json')
                self.end_headers()
                self.wfile.write("BAD REQUEST - empty payload", "utf8")
                return
            data = unquote(data.decode())
            # print ("<debug>");
            # print (data);
            # print ("</debug>")
            jsonData = json.loads(data)
            bytes_image = base64.b64decode(jsonData['image'])
            img = self.server.img_utils.dataToImage(bytes_image)
            result=self.server.model.evaluate(img)
            # genero una imagen segmentada segun las detecciones de la red
            mask_image = self.server.img_utils.generateMaskImage(img, result)
            self.server.img_utils.saveImage(mask_image,'test_'+str(time.time())+'.jpg')
            dataResponse = {}
            # Send message back to client
            dataResponse["message"] = "Hello POST!"
            dataResponse["result"] = self.server.img_utils.predictions_array(result)
            self.do_response(dataResponse)
    # GET
    def do_GET(self):
        if self.path == '/position':
            self.do_GET_position()
        else:
            self.do_GET_env()

    def do_GET_position(self):
        self.do_response(getPosition())

    def do_GET_env(self):
        dataResponse = {}
        # Send message back to client
        dataResponse['message'] = 'Hello POST!'
        dataResponse['dimensiones'] = {'ancho': 4.00, 'alto': 3.00, 'largo': 6.00}
        dataResponse['persona'] = resetPosition()
        # dataResponse['persona'] = {'x': 2.90, 'y': 1.50, 'z': 2.30}
        dataResponse['objetos'] = []
        dataResponse['objetos'].append(createObject(50,55,200,100,110,200, 'armario')) # armario bajo
        dataResponse['objetos'].append(createObject(260,25,275,75,50,50, 'baul')) # baul
        dataResponse['objetos'].append(createObject(240,40,150,75,80,100, 'mesa1')) # mesa
        dataResponse['objetos'].append(createObject(375,100,150,50,200,150, 'cama')) # cama rebatible

        dataResponse['objetos'].append(createObject(50,150,310,100,300,20, 'paredA')) # pared intermedia A
        dataResponse['objetos'].append(createObject(150,250,310,100,100,20, 'paredARCO')) # pared intermedia ARCO
        dataResponse['objetos'].append(createObject(300,150,310,200,300,20, 'paredB')) # pared intermedia B

        dataResponse['objetos'].append(createObject(125,50,562,100,100,75, 'mesa2')) # mesa 
        dataResponse['objetos'].append(createObject(25,50,410,50,100,180, 'mesada')) # mesada
        dataResponse['objetos'].append(createObject(225,55,370,50,110,100, 'desayunador')) # desayunador
        dataResponse['objetos'].append(createObject(375,40,560,50,80,80, 'mesita')) # mesita

        dataResponse['objetos'].append(createObject(350,150,420,100,300,200, 'EspacioCerrado')) # Espacio cerrado

        self.do_response(dataResponse)

    def do_response(self,dataResponse):
        # Send response status code
        self.send_response(200)
        # Send headers
        self.send_header('Content-type','application/json')
        self.end_headers()
        # Write JSON response as utf-8 data
        # response = json.dumps(dataResponse,indent=4,sort_keys=True,cls=NumpyArrayEncoder)
        response = json.dumps(dataResponse, indent=2,cls=NumpyArrayEncoder)
        print('response: '+ response)
        self.wfile.write(bytes(response, "utf8"))