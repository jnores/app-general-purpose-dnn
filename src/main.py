from pytorchUtils import DNN_Model
import imageUtils
from fileUtils import FileUtils
import time

from configFile import *


def process_image(model,img_name,input_folder='in/',output_folder='out/'):
    if (imageUtils.isNotImage(img_name)):
        print('no es imagen. ignoro el archivo')
        return
    # identifico el nombre de la imagen para generar salidas homonimas
    file_name=img_name.split('/')[-1].split('.')[0]
    imgFileResult = output_folder+file_name+'.jpg'
    txtFileResult = output_folder+file_name+'.txt'
    # abro la imagen
    img = imageUtils.load_image(input_folder+img_name)
    # proceso la imagen con la red neuronal profunda
    result=model.evaluate(img)
    # genero una imagen segmentada segun las detecciones de la red
    mask_image = imageUtils.generateMaskImage(img, result)
    # guardo la imagen segmentada y el detalle de la deteccion.
    imageUtils.saveImage(mask_image,imgFileResult)
    imageUtils.saveDetail(result,txtFileResult)
    # muestro las imagenes por pantalla
    #imageUtils.show_image(mask_image)

def procesar_archivos(model,files_list):
    for file_name in files_list:
        print('INICIO - Procesando: ', file_name)
        init_time = time.time()
        process_image(model,file_name)
        end_time = time.time()
        print(f'FIN    - tiempo: {end_time - init_time:.3f}s')


# primero creo el modelo y lo cargo en memoria
model=DNN_Model(THRESHOLD)
fileUtils = FileUtils(FB_CREDENTIALS_PATH,BUCKET_NAME)
# img_name='Photo.png'
# process_image(model, img_name)

# 2do inicio el loop infinito para procesar las imagenes de FB
# '''
while True:
    fileUtils.sync_in_folder()
    files_list = fileUtils.lista_archivos_no_procesados()
    if (len(files_list)>0):
        procesar_archivos(model, files_list)
        fileUtils.sync_out_folder()
    else:
        print('no hay archivos por procesar. dormir: ',SLEEP_TIME)
        time.sleep(SLEEP_TIME)
# '''