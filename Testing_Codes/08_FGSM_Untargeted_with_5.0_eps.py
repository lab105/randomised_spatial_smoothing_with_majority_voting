import warnings
warnings.filterwarnings('ignore')
import time
import csv

#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

# Disable TensorFlow eager execution:
import tensorflow as tf
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# Load Keras dependencies:
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

# Load ART dependencies:
from art.estimators.classification import KerasClassifier
from art.preprocessing.preprocessing import Preprocessor
from art.defences.preprocessor import SpatialSmoothing
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.utils import to_categorical


from imagenet_stubs.imagenet_2012_labels import name_to_label, label_to_name

#Loading the labels file and getting them in correct format
with open('imagepaths.txt', 'r') as text:
    imagepaths = text.read()[:-1].split('\n')
with open('LOC_synset_mapping.txt','r') as text:
    imagelabels= text.read()[:-1].split('\n')

#Appending labels to a list in accordance of the loaded images    
image_labels=list()
for i in imagepaths:
    for j in imagelabels:
        if i[i.rfind('/')+1:i.rfind('_')]==j[:j.find(' ')]:
            image_labels.append(name_to_label(j[j.find(' ')+1:]))

#Loading the images
images_list = list()
for i,image_path in enumerate(imagepaths):
    #Specify i to be the number of samples to be tested on
    while i<1000:
        im = image.load_img(image_path, target_size=(224, 224))
        im = image.img_to_array(im)
        images_list.append(im)
        break
images = np.array(images_list)

model = ResNet50(weights='imagenet')

class ResNet50Preprocessor(Preprocessor):

    def __call__(self, x, y=None):
        return preprocess_input(x.copy()), y

    def estimate_gradient(self, x, gradient):
        return gradient[..., ::-1] 
    
preprocessor = ResNet50Preprocessor()

normalclassifier=KerasClassifier(model,clip_values=(0, 255), preprocessing=preprocessor)

class BetterClassifierWithoutRandom(KerasClassifier):
        
    def predict(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create containers for our predictions and spatial smoothening window sizes
        prediction_labels=[]
        prediction_scores=[]
        label_counts={}
        smoothening_values=[1,2,3,3,4,4,5,5,6,7]
        
        #Predict with each window size, and store the labels and prediction scores in their containers
        for i in smoothening_values:
            ss=SpatialSmoothing(window_size=i)
            x_def,_=ss(x_preprocessed)
            pred = self._model.predict(x_def,batch_size=batch_size)
            label = label_to_name(np.argmax(pred, axis=1)[0])
            prediction_scores.append(pred)
            prediction_labels.append(label)
        
        #finding out which label is most frequently identified, and taking a weighted mean of that label's scores
        for label in prediction_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
                
        most_frequent_label = max(label_counts, key=label_counts.get)
        most_frequent_indexes = [i for i, p in enumerate(prediction_labels) if p == most_frequent_label]
        most_frequent_pred_scores = [prediction_scores[i] for i in most_frequent_indexes]

        weights = 1/np.array([smoothening_values[i] for i in most_frequent_indexes])
        predictions = np.average(most_frequent_pred_scores,axis=0,weights=weights)
        
        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions
    
betterclassifierwithoutrandom = BetterClassifierWithoutRandom(model,clip_values=(0, 255), preprocessing=preprocessor)

class BetterClassifier(KerasClassifier):
        
    def predict(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Create containers for our predictions and spatial smoothening window sizes
        prediction_labels=[]
        prediction_scores=[]
        label_counts={}
        smoothening_values=[np.random.randint(1,7) for i in range(10)]
        
        #Predict with each window size, and store the labels and prediction scores in their containers
        for i in smoothening_values:
            ss=SpatialSmoothing(window_size=i)
            x_def,_=ss(x_preprocessed)
            pred = self._model.predict(x_def,batch_size=batch_size)
            label = label_to_name(np.argmax(pred, axis=1)[0])
            prediction_scores.append(pred)
            prediction_labels.append(label)
        
        #finding out which label is most frequently identified, and taking a weighted mean of that label's scores
        for label in prediction_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
                
        most_frequent_label = max(label_counts, key=label_counts.get)
        most_frequent_indexes = [i for i, p in enumerate(prediction_labels) if p == most_frequent_label]
        most_frequent_pred_scores = [prediction_scores[i] for i in most_frequent_indexes]

        weights = 1/np.array([smoothening_values[i] for i in most_frequent_indexes])
        predictions = np.average(most_frequent_pred_scores,axis=0,weights=weights)
        
        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)
        return predictions
    
betterclassifier = BetterClassifier(model,clip_values=(0, 255), preprocessing=preprocessor)

targeted=False
target_label=94
att="08_FGSM_Untargeted_with_5.0_eps"
attack1=FastGradientMethod(normalclassifier, eps=5.0)
attack2=FastGradientMethod(betterclassifierwithoutrandom, eps=5.0)
attack3=FastGradientMethod(betterclassifier, eps=5.0)


def test(attack1,attack2,attack3,imageset):
    totaltime=time.time()
    def test_part1(attack,classifier,targeted):
        if targeted==True:
            x_art_adv = attack.generate(x_art,y=to_categorical([target_label]))
        else: x_art_adv = attack.generate(x_art)
        time1=time.time()
        pred1=classifier.predict(x_art_adv)
        time1=time.time() - time1
        label1=np.argmax(pred1,axis=1)[0]
        confidence1=pred1[:,label1][0]
        l_0_1 = int(99*len(np.where(np.abs(x_art[0] - x_art_adv[0])>0.5)[0]) / (224*224*3)) + 1   
        l_1_1 = int(99*np.sum(np.abs(x_art[0] - x_art_adv[0])) / np.sum(np.abs(x_art[0]))) + 1
        l_2_1 = int(99*np.linalg.norm(x_art[0] - x_art_adv[0]) / np.linalg.norm(x_art[0])) + 1 
        l_inf_1 = int(99*np.max(np.abs(x_art[0] - x_art_adv[0])) / 255) + 1
        if image_labels[i] == label1:
            result1=1
        elif target_label == label1 and targeted==True:
            result1=-1
        else: result1=0

        return[label1,'{0:.2f}'.format(confidence1),l_0_1,l_1_1,l_2_1,l_inf_1,result1,time1]
    
    with open(f'{att}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['INPUT_IMG','NORM_CLAS_ADV_IMG','NCAI_CONF','NCAI_L_0','NCAI_L_1','NCAI_L_2','NCAI_L_INF','NCAI_RSLT','NCAI_TIME','BET_CLAS_NO_RAND_ADV_IMG','BCNRAI_CONF','BCNRAI_L_0','BCNRAI_L_1','BCNRAI_L_2','BCNRAI_L_INF','BCNRAI_RSLT','BCNRAI_TIME','BET_CLAS_ADV_IMG','BCAI_CONF','BCAI_L_0','BCAI_L_1','BCAI_L_2','BCAI_L_INF','BCAI_RSLT','BCAI_TIME'])
    for i,image_label in enumerate(imageset):
        #while i>checkpoint:
            x_art = np.expand_dims(imageset[i], axis=0)
            with open(f'{att}.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_labels[i]]+test_part1(attack1,normalclassifier,targeted)+test_part1(attack2,betterclassifierwithoutrandom,targeted)+test_part1(attack3,betterclassifier,targeted))
            
            print(i)
            #break
    totaltime=time.time() - totaltime
    print('Total time taken for attack =',totaltime,'seconds')
    
    
test(attack1,attack2,attack3,images)