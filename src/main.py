from PIL import Image
from pytorchUtils import DNN_Model
from imageUtils import saveDetail,saveImage,generateMaskImage
import matplotlib.pyplot as plt


def process_image(model,img_name,input_folder='in/',output_folder='out/'):
    # identifico el nombre de la imagen para generar salidas homonimas
    file_name=img_name.split('/')[-1].split('.')[0]
    imgFileResult = output_folder+file_name+'.jpg'
    txtFileResult = output_folder+file_name+'.txt'
    # abro la imagen
    img = Image.open(input_folder+img_name).convert("RGB")
    # proceso la imagen con la red neuronal profunda
    result=model.evaluate(img)
    # genero una imagen segmentada segun las detecciones de la red
    mask_image = generateMaskImage(img, result)
    # guardo la imagen segmentada y el detalle de la deteccion.
    saveImage(mask_image,imgFileResult)
    saveDetail(result,txtFileResult)
    plt.figure(figsize=(6,4))
    plt.imshow(img)
    print("IMAGE 1")
    plt.show()
    plt.figure(figsize=(6,4))
    plt.imshow(mask_image)
    print("IMAGE 2")
    plt.show()

    



model=DNN_Model(.1)
img_name='Photo.png'
process_image(model, img_name)