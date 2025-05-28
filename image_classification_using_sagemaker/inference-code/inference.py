import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import io
import logging
import smdebug
import smdebug.pytorch as smd
import json

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#load the model
def model_fn(model_dir):
    logger.info(f"Entering model_fn")
    #load the model from the directory
    model = torch.load(f'{model_dir}/model.pth')
    #set the mode of the model to eval
    model.eval()
    return model

#process the input
def input_fn(request_body, content_type):
    logger.info(f"Entering input_fn")
    if content_type == 'image/jpeg':
        image = Image.open(io.BytesIO(request_body))
        
        #apply the needed transformation. these are the same that we applited while training the model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        ])
        #change the image to a vector of size 1 and return
        return transform(image).unsqueeze(0)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
        
#process the output of the model
def predict_fn(input_data, model):
    logger.info(f"Entering predict_fn")
    with torch.no_grad():
        output = model(input_data)
        probabilities = F.softmax(output, dim=1)
        confidence, class_index = torch.max(probabilities, 1)
    return {"class": class_index.item(), "confidence": confidence.item() * 100}

#format the output
def output_fn(prediction, content_type):
    logger.info(f"Entering output_fn")
    return json.dumps({
            "predicted_class": prediction['class'],
            "confidence": f"{prediction['confidence']:.2f}%"
        })