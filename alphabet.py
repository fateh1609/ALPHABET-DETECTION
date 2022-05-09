from threading import ExceptHookArgs
from turtle import resizemode
import cv2 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

#setting up https context to fetch data
if (not os.environ.get('PYTHONHTTPSVERIFY','')and
   getattr(ssl,'_create_unverified_context',None)):
   ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 9, train_size = 7500,test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(X_train_scaled,y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame=cap.read()

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        height,width = gray.shape
        upper_left = (int(width/2-56),int(height/2-56))
        botom_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray,upper_left,botom_right,(0,255,0),2)

        # roi = region of interest
        roi = gray[upper_left[1]:botom_right[1],upper_left[0]:botom_right[0]]

        #cv2 -> pil format
        im_pil = Image.fromarray(roi)

        # convert t0 grey scale image - "L" format means each pixel is reprented by a single value 0->255
        image_vw = im_pil.convert("L")
        image_vw_resized = image_vw.resize((28,28),Image.ANTIALIAS)

        #invert the image
        image_vw_resized_inverted = PIL.ImageOps.invert(image_vw_resized)
        pixel_filter = 20
        min_pixel = np.percentile(image_vw_resized_inverted,pixel_filter)
        image_vw_resized_inverted_scaled = np.clip(image_vw_resized_inverted - min_pixel,0,255)
        max_pixel = np.max(image_vw_resized_inverted)
        
        #converting to an array
        image_vw_resized_inverted_scaled = np.asarray(image_vw_resized_inverted_scaled)/max_pixel
        test_sample = np.array(image_vw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("predicted classes: " , test_pred)

        #display the resulting frame
        cv2.imshow("frame",gray)
        if cv2.waitKey(1)& 0xFF == ord("q"):
            break
    except Exception as e:
        pass
cap.release()
cv2.destroyAllWindows()