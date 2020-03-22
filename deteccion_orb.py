import numpy as np
import cv2
from matplotlib import pyplot as plt

imagenes = []
kp = []
dc = []
for i in range(48):
    nombre = "train/frontal_"+str(i+1)+".jpg"
    img = cv2.imread(nombre, 0)
    imagenes.append(img)

sift = cv2.ORB_create(500, 1.3, 4, 31, 0, 2, 100, 31, 20)

keypts = sift.detect(imagenes[1])
frame = cv2.drawKeypoints(imagenes[1], keypts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('frame', frame)
cv2.waitKey(0)
for i in range(48):
    kp1, des1 = sift.detectAndCompute(imagenes[i], None)
    kp.append(kp1)
    dc.append(des1)







