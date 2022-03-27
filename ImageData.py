
#!pip3 install opencv-python
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random

#!pip install scikit-image
#from skimage import data, color

def Load_Image_Dataset(img_folder):
   img_data_array=[]
   class_id=[]
   for dirl in os.listdir(img_folder):
       for file in os.listdir(os.path.join(img_folder,dirl)):
      
           if any([file.endswith(x) for x in ['.png', '.jpeg', '.jpg']]):
                image_path=os.path.join(img_folder,dirl,file)
                image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
             
                img_data_array.append(image)
                class_id.append(int(dirl))
   return img_data_array,class_id

def show_image(image, title='Image', cmap_type='gray'):
    plt.rcParams["figure.figsize"] = [4, 3]
    plt.imshow(image,cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

def ShowRandomImages(X_test,y_pred):
    n_rows = 3
    n_cols = 15
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            id = random.randrange(1,len(X_test))
            plt.imshow(X_test[id].reshape(48,48),cmap='gray')
            plt.axis('off')
            plt.title(labelType(y_pred[id]), fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()

def extractfeatures(imagedataset):
    Features = []
    for image in imagedataset:
        _feature = np.reshape(image, 48*48)
        Features.append(_feature)
    return Features

def labelType(argument):
    switcher = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
        "angry": "Angry",
        "disgust": "Disgust",
        "fear": "Fear",
        "happy": "Happy",
        "sad": "Sad",
        "surprise": "Surprise",
        "neutral": "Neutral"
    }
    return switcher.get(argument, "nothing")


## Dimention Reduction using PCA for linear data seperation
############################################################
from sklearn.decomposition import PCA

def PCA_FeatureReduction(X_input , featuresnumber , OriginalDataforShape, showimages=1):
    pca = PCA(n_components=featuresnumber, random_state=22) ## get 1000 features
    pca.fit(X_input)
    X = pca.transform(X_input)
    print("Original number of features : ",np.array(X_input).shape)
    print("Reduced number of features : ",X.shape)
    
    if (showimages == 1):
        n=500 # randome image
        # show Original Image
        show_image(X_input[n].reshape(np.array(OriginalDataforShape).shape[1], np.array(OriginalDataforShape).shape[2]),title="Original Image")
        # show image after feature reduction
        approximation = pca.inverse_transform(X)
        show_image(approximation[n].reshape(np.array(OriginalDataforShape).shape[1], np.array(OriginalDataforShape).shape[2]),title="Reduced Image")
        
    return X








