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

class RequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.server.work_lock.locked():
            msg = "LOCKED"
        else:
            msg = "UNLOCKED"
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(bytes(msg, "utf-8"))

    # POST
    def do_POST(self):
        # Procesar request
        if self.server.work_lock.locked():
            self.send_response(503)
            self.wfile.write("SERVER BUSY", "utf8")
            return
        with self.server.work_lock:
            time.sleep(5)
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
            # Send response status code
            self.send_response(200)
            # Send headers
            self.send_header('Content-type','application/json')
            self.end_headers()
            # Write JSON response as utf-8 data
            # response = json.dumps(dataResponse,indent=4,sort_keys=True,cls=NumpyArrayEncoder)
            response = json.dumps(dataResponse,cls=NumpyArrayEncoder)
            # print('response: '+ response)
            self.wfile.write(bytes(response, "utf8"))