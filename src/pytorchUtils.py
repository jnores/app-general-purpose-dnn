import torch
import torchvision
from torchvision import transforms as transforms
from coco_names import  COCO_INSTANCE_CATEGORY_NAMES as COCO_NAMES


class DNN_Model:
    def __init__(self,threshold=0.8):
        self.model =  torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device).eval()
        self.score_threshold=threshold
        self.transform = transforms.Compose([transforms.ToTensor()])

    def evaluate(self,image):
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image)
        
        scores = prediction[0]['scores'].cpu()
        labels = prediction[0]['labels'].cpu()
        # masks = prediction[0]['masks'].cpu() # Este deja valores entre 0 y 1
        masks = (prediction[0]['masks']>0.5).cpu() # este lo convierte en una matriz booleana
        boxes = prediction[0]['boxes'].cpu()
        detect_ok_condition = scores >=self.score_threshold
        scores = scores[detect_ok_condition].numpy()
        labels = labels[detect_ok_condition].numpy()
        masks = masks[detect_ok_condition].numpy()
        boxes = boxes[detect_ok_condition].numpy()
        predictions = []
        for i in range(len(scores)):
            predictions.append({
                'score': scores[i],
                'label': labels[i],
                'name': COCO_NAMES[labels[i]],
                'mask': masks[i,0],
                'box': [(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]) ], # (x1,y1),(x2,y2)
                'color':'undefined_color'
                })
        return predictions
