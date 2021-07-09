import numpy as np 
from tensorflow.keras.models import load_model
import cv2

cap = cv2.VideoCapture(0)

model = load_model('detectorModel.h5')

while True:
    _, frame = cap.read()
    curFrame = np.array(frame)
    curFrame = cv2.resize(curFrame, (32, 32))
    curFrame = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)
    curFrame = curFrame.reshape(1, 32, 32, 1)
    curFrame = curFrame/255

    classIndex = int(model.predict_classes(curFrame))
    print(classIndex)
    cv2.putText(frame, str(classIndex), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break