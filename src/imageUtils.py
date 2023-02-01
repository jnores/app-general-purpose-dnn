import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import io

# creo un color aleatorio para cada clase/categoria que identifica la red
COLORS = np.random.uniform(0,127,size=(100,3))

VALID_IMAGE_EXTENSIONS=['jpg','png','jpeg']

def isNotImage(name):
    ext=name.split('/')[-1].split('.')[-1]
    return not ext in VALID_IMAGE_EXTENSIONS

def load_image(file_name):
    return Image.open(file_name).convert("RGB")

def dataToImage(bytes_image):
    stream = io.BytesIO(bytes_image)
    return Image.open(stream).convert("RGB")


def generateMaskImage(image,predictions):
    alpha = 1
    mask_opacity = .4
    gamma = 0
    mask_image = np.zeros_like(image)
    for pred in predictions:
        red_map = np.zeros_like(pred['mask']).astype(np.uint8)
        green_map = np.zeros_like(pred['mask']).astype(np.uint8)
        blue_map = np.zeros_like(pred['mask']).astype(np.uint8)
        # marca los colores en las posiciones correctas de la matriz de la imagen.
        color = COLORS[random.randrange(0,len(COLORS))]
        red_map[pred['mask']==True], green_map[pred['mask']==True], blue_map[pred['mask']==True] = color
        pred['color'] = f'rgb({int(color[0])} {int(color[1])} {int(color[2])})'
        #genera una pseudo-imagen con la mascara
        segmentation_map = np.stack([red_map,green_map, blue_map], axis=2)
        # Agrega la mascara sobre la imagen
        mask_image=cv2.cvtColor(mask_image,cv2.COLOR_RGB2BGR)
        cv2.addWeighted(mask_image,alpha,segmentation_map,mask_opacity,gamma,mask_image)

    return mask_image

def show_image(img):
    plt.figure(figsize=(6,4))
    plt.imshow(img)
    plt.show()

    
def saveImage(image,file_name='out.jpg'):
    cv2.imwrite(file_name,image)


def saveDetail(predictions,file_name='out.txt'):
    with open(file_name, 'w') as f:
        for pred in predictions:
            f.write(predToString(pred))
            f.write('\n')

def predToString(prediction):
    line=[
        str(prediction['score']),
        str(prediction['label']),
        str(prediction['name']),
        str(prediction['color']),
        str(prediction['box'])
    ]
    return ';'.join(line)
    

def predictions_array(predictions):
    # print("Predictions: ")
    # print(predictions)

    return [{
                'score': float(p['score']),
                'label': int(p['label']),
                'name': p['name'],
                'mask': getMask(p['mask']),
                # 'box': [p['box'][0][0],p['box'][0][1],p['box'][1][0],p['box'][1][1]], # (x1,y1),(x2,y2)
                }
            for p in predictions]

def getMask(mtx):
    #print("matrix: ")
    #print(mtx.shape)
    new = mtx.reshape(-1)
    #print (new.shape)
    return new;