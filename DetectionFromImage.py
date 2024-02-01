import cv2
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms
from torchvision.models.detection import _utils
from torchvision.models.detection.ssd import SSDClassificationHead

IMAGE = 'photo.jpg'

MODEL_FILE = 'best_custom.pth'
CLASSES = ['__background__', 'one', 'two', 'three']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SIZE = 640

weights = SSD300_VGG16_Weights.COCO_V1
model = ssd300_vgg16(weights=weights)
in_channels = _utils.retrieve_out_channels(model.backbone, (SIZE, SIZE))
num_anchors = model.anchor_generator.num_anchors_per_location()
model.head.classification_head = SSDClassificationHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=NUM_CLASSES)
model.transform.min_size = (SIZE,)
model.transform.max_size = SIZE

checkpoint = torch.load(MODEL_FILE, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

preprocess = weights.transforms()
transform = transforms.ToTensor()

cv_image = cv2.imread(IMAGE)
rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
torch_image = transform(rgb_image)
batch = preprocess(torch_image).unsqueeze(0)

prediction = model(batch)[0]

labels = [CLASSES[i] for i in prediction["labels"]]

for i in range(len(labels)):
    score = prediction['scores'][i].item()
    if score > 0.7:
        x1, y1, x2, y2 = prediction['boxes'][i] 
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = labels[i]
        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(cv_image, f'{label}: {score:.4f}', (x1,y1+25), 1, 2, (255, 0 , 255), 2)

cv2.imshow('Result', cv_image)
cv2.waitKey(0)