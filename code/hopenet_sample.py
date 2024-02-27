import datetime
#    Testing pre-trained hopenet on images from a folder
#    using hopenet_robust_alpha1.pkl

import os

import PIL.Image
import cv2
import numpy as np
from PIL.Image import Image

# import tensorflow
from mtcnn import MTCNN

import torch
import torchvision
from torch.backends import cudnn
from torchvision import transforms
from torch.autograd import Variable

from hopenet import hopenet
from hopenet import utils

def load_face(path, resize=None):
    '''
    Loads an image from the input path and runs face detection
    :param path: path of the input image wit file name
    :param resize: default None, if (x,y) tuple the return image will be resized to (x,y)
    :return: image of the detected face or None if no face is found
    '''

    print(path)

    with PIL.Image.open(path) as item:
        if item is None:
            return

        # TODO: create a class and use a singleton for memory use
        detector = MTCNN()

        cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        #cv2.imshow("one_item", cv_img)
        #cv2.waitKey(0)

        # detect face
        faces = detector.detect_faces(cv_img)
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            print(x, y, w, h)

            face_img = item.crop((int(x-20), int(y-20), int(x+w+20), int(y+h+20)))
            face_img = face_img.convert("RGB")

            return face_img

def hopenet_predict(model, face_image, save_axis = False):
    '''
    Takes an input image with loaded model, resizes the image and gets the 3 Euler angles for the input face
    It is assumed that the input image is already run through face detection
    :param model: configured and loaded hopenet model
    :param face_image: input image for which the prediction should run
    :param save_axis: if set to true the result axis will be added to the image and saved to the images folder
    :return: yaw, pitch and roll real values (Euler angles) in (yaw, pitch, roll) format
    '''

    if save_axis:
        cv2_img = np.asarray(face_image)
        cv2_img = cv2.resize(cv2_img, (224, 224))[:, :, ::-1]
        cv2_img = cv2_img.astype(np.uint8).copy()

    face_image = transformations(face_image)
    face_image = face_image.unsqueeze(0)
    face_images = Variable(face_image)
    yaw, pitch, roll = model(face_images)

    # results are now tensors, using calculation from the original hopenet implementation
    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
    roll_predicted = utils.softmax_temperature(roll.data, 1)

    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

    pitch = pitch_predicted[0]
    yaw = -yaw_predicted[0]
    roll = roll_predicted[0]

    if save_axis:
        utils.draw_axis(cv2_img, yaw, pitch, roll, size=100)
        cv2.imwrite(f'{eval_path}images/{str(datetime.datetime.now())}.jpg', cv2_img)

    return yaw, pitch, roll

def load_image_names(path):
    '''
    Loads jpg files found in the input folder. Not recursive
    :param path: input path
    :return: list of file names with jpg extensions
    '''
    jpgs = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpg')]

    return jpgs


cudnn.enabled = True
# running on cpu only now
gpu = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

snapshot_path = "../models/hopenet_robust_alpha1.pkl"
eval_path = "/Users/laszlokovari/Documents/Prog/posture/"

# ResNet50 structure
# for details on resnet50 see https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

# load snapshot into model
save_saved_dict = torch.load(snapshot_path, map_location=device)
model.load_state_dict(save_saved_dict)

transformations = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# model.cuda(gpu)
# testing on M1 mac
model.to("mps")

# model is set up, can test
model.eval()

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor).to(device)

# loading image list from eval path
snapshots = load_image_names(eval_path)

for snapshot in snapshots:
    face_detected = load_face(snapshot)

    if face_detected is not None:
        Image.show(face_detected)

        result = hopenet_predict(model, face_detected, True)
        print(result)

