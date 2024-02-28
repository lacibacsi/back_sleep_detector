import os
from datetime import datetime

import PIL
import cv2
import torch
import torchvision
from PIL.Image import Image
from mtcnn import MTCNN
from torch.backends import cudnn
from torchvision import transforms
from torch.autograd import Variable

from hopenet import hopenet
import util_functions
from hopenet import utils

# imports for training
import pandas as pd
import numpy as np
from sklearn import metrics, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BackSleepDetector:
    """
        Sample class for training and inference of images on detecting if the person is laying on their back
        Usage:
        - call train with the folder where all the images are stored in the following structure:
            - images
            -   true ->     images of laying face orientation
            -   false ->    images of not laying face orientation
            train can also accept a correction angle (yaw, pitch, roll) for the camera,
            the default training mode is when the camera is directly above the person

            the training uses hopenet pre-trained model hopenet_robust_alpha1.pkl.
            For examples and details see hopenet_sample.py
            default train-test is 70-30
            classification model is: <TBD>
            the model weights are stored in the model path input

        - call predict with a single image path
            - predict will use the model path and return a simple boolean, True: back laying, False: not on back
    """

    def __init__(self):
        # default model path
        self._model_path = '../models'
        self._regression_model = self._model_path + "/back_detector.pkl"
        self._parsed_training_data = self._model_path + "/training_data.pkl"
        self._snapshot_path = "../models/hopenet_robust_alpha1.pkl"
        self._eval_path = "/Users/laszlokovari/Documents/Prog/posture/"

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

        self._transformations = transforms.Compose(
            [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        self._idx_tensor = [idx for idx in range(66)]
        self._idx_tensor = torch.FloatTensor(self._idx_tensor).to(self._device)

        self._face_detector = MTCNN()

        self._setup_hopenet_model()

    def _setup_hopenet_model(self):
        """
        Sets up the hopenet model and loads the saves snapshot (pretrained) in it
        :return: None
        """

        cudnn.enabled = True
        # ResNet50 structure
        # for details on resnet50 see https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758
        self._hopenet_model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        # load snapshot into model
        self._save_saved_dict = torch.load(self._snapshot_path, map_location=self._device)
        self._hopenet_model.load_state_dict(self._save_saved_dict)

        self._transformations = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        self._hopenet_model.to("mps")
        self._hopenet_model.eval()

    def load_dataset(self, training_data_path):
        """
        loads the data set image names, sets up a dataframe and returns with the target value and filename set
        :param training_data_path:
        :return: dataframe with filename and class set, yaw, pitch, roll left empty
        """

        assert training_data_path is not None

        # reading true values
        true_files = util_functions.load_image_names(training_data_path + '/true/')
        false_files = util_functions.load_image_names(training_data_path + '/false/')

        df = pd.DataFrame(columns=['filename', 'yaw', 'pitch', 'roll', 'target' ])

        true_rows = [{'filename': file,'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'target': 1} for file in true_files]
        false_rows = [{'filename': file, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'target': 0} for file in false_files]

        df = pd.concat([df, pd.DataFrame(true_rows)], ignore_index=True)
        df = pd.concat([df, pd.DataFrame(false_rows)], ignore_index=True)
        return df

    def load_face(self, path, resize=None):
        '''
        Loads an image from the input path and runs face detection
        :param path: path of the input image wit file name
        :param resize: default None, if (x,y) tuple the return image will be resized to (x,y)
        :return: image of the detected face or None if no face is found
        '''

        with PIL.Image.open(path) as item:
            if item is None:
                return

            cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            # detect face
            faces = self._face_detector.detect_faces(cv_img)
            if len(faces) > 0:
                x, y, w, h = faces[0]['box']
                print(x, y, w, h)

                face_img = item.crop((int(x - 20), int(y - 20), int(x + w + 20), int(y + h + 20)))
                face_img = face_img.convert("RGB")

                return face_img

    def setup_hopenet(self):
        """
        Sets up the hopenet model and returns it
        :return: Hopenet model with pretrained values loaded, ready for prediction
        """

        # testing on M1 mac


    def hopenet_predict(self, face_image, save_axis=False):
        '''
        Takes an input image with loaded model, resizes the image and gets the 3 Euler angles for the input face
        It is assumed that the input image is already run through face detection
        :param face_image: input image for which the prediction should run
        :param save_axis: if set to true the result axis will be added to the image and saved to the images folder
        :return: yaw, pitch and roll real values (Euler angles) in (yaw, pitch, roll) format
        '''

        if save_axis:
            cv2_img = np.asarray(face_image)
            cv2_img = cv2.resize(cv2_img, (224, 224))[:, :, ::-1]
            cv2_img = cv2_img.astype(np.uint8).copy()

        face_image = self._transformations(face_image)
        face_image = face_image.unsqueeze(0)
        face_images = Variable(face_image)
        yaw, pitch, roll = self._hopenet_model(face_images)

        # results are now tensors, using calculation from the original hopenet implementation
        # Binned predictions
        _, yaw_bpred = torch.max(yaw.data, 1)
        _, pitch_bpred = torch.max(pitch.data, 1)
        _, roll_bpred = torch.max(roll.data, 1)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * self._idx_tensor, 1).cpu() * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * self._idx_tensor, 1).cpu() * 3 - 99
        roll_predicted = torch.sum(roll_predicted * self._idx_tensor, 1).cpu() * 3 - 99

        pitch = pitch_predicted[0]
        yaw = -yaw_predicted[0]
        roll = roll_predicted[0]

        if save_axis:
            utils.draw_axis(cv2_img, yaw, pitch, roll, size=100)
            cv2.imwrite(f'{self._eval_path}images/{str(datetime.now())}.jpg', cv2_img)

        return yaw, pitch, roll

    def train(self, training_data_path, save_model=True, use_saved_data =True, camera_angle_correction=(0.0, 0.0, 0.0)):
        """
        Trains a simple nearest neighbor model based on a pretrained hopenet model and the input images
        The input parameters to the model are the yaw, pitch, roll values from hopenet and the dependent variable
        is the binary classification whether the person in the image is laying on their back or not
        :param training_data_path: the root folder of training images. It's expected to have true and false sub-folders
        :param save_model: indicates if the resulting trained model should be saved to the model path
        :param use_saved_data: indicates if the previously save face data should be loaded and reused
        :param camera_angle_correction: if the images taken were taken by a camera whose angle needs to be offset
        :return: None
        """

        df_alldata = None
        data_size = None

        if use_saved_data and os.path.isfile(self._parsed_training_data):
                df_alldata = pd.read_pickle(self._parsed_training_data)
                data_size = len(df_alldata)
                print(f'loaded previously saved {data_size} images')
        else:
            df_alldata = self.load_dataset(training_data_path)

            data_size = len(df_alldata)
            print(f'{data_size} images loaded')

            # for each image the yaw, pitch, roll values need to be predicted from the hopenet model
            # the model is already set up

            # TODO: iterrows is slow, use itertuples for really large datasets
            counter = 0
            for index, row in df_alldata.iterrows():
                face_detected = self.load_face(row['filename'])
                if face_detected is not None:
                    #Image.show(face_detected)

                    yaw, pitch, roll = self.hopenet_predict(face_detected, False)
                    # updating the rows
                    df_alldata.loc[index, 'yaw'] = yaw.item()
                    df_alldata.loc[index, 'pitch'] = pitch.item()
                    df_alldata.loc[index, 'roll'] = roll.item()

                counter = counter + 1
                if counter % 10 == 0:
                    print(f'{counter} item parsed from {data_size}')

            print(df_alldata.head())

            print(f'all {data_size} image yaw, pitch, and roll values calculated.')

            if save_model:
                df_alldata.to_pickle(self._parsed_training_data)
                print('parsed image data saved')

        print(f'starting knn classification training')

        x = df_alldata[['yaw','pitch','roll']]
        y = df_alldata[['target']].astype(int)

        # splitting the data set into train / test / validation sets
        # using 0.3 for testing and shuffle true randomize
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

        clf = neighbors.KNeighborsClassifier()
        clf.fit(x_train, y_train)

        # evaluate performance
        score = clf.score(x_test, y_test)
        print(f'k nearest neighbor model trained on {len(y_train)} images, resulting score on {len(y_test)} is {score} ')

        # further metrics
        y_pred = pd.Series(clf.predict(x_test))
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred))

        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print(cnf_matrix)


if __name__ == '__main__':
    detector = BackSleepDetector()
    detector.train("../dataset/headpose")


