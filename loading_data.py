from keras.models import load_model
import cv2
import numpy as np

model = load_model('./training/model-012.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap=cv2.VideoCapture(0)


labels_dict={0:'Angry',1:'Neutral',2:'Sad',3:"happy"}
color_dict={0:(0,0,255),1:(0,255,0),2:(255,0,0),3:(255,0,255)}


while(True):

    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,3)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
    if ret==False:
        break
        
cv2.destroyAllWindows()
cap.release()
