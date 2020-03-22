import numpy as np
import cv2
from matplotlib import pyplot as plt

imagenes = []
kp = []
dc = []
good = []
for i in range(48):
    nombre = "train/frontal_"+str(i+1)+".jpg"
    img = cv2.imread(nombre, 0)
    imagenes.append(img)

sift = cv2.ORB_create(500, 1.3, 4, 31, 0, 2, 100, 31, 20)

#keypts = sift.detect(imagenes[1])
#frame = cv2.drawKeypoints(imagenes[1], keypts, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('frame', frame)
#cv2.waitKey(0)
for i in range(48):
    kp1, des1 = sift.detectAndCompute(imagenes[i], None)
    kp.append(kp1)
    dc.append(des1)

for j in range(48):
    # BFMatcher with default params
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(dc[0], dc[j], k=2)
    # Apply ratio test

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    if j==47:
        img3 = cv2.drawMatchesKnn(imagenes[0], kp[0], imagenes[j], kp[j], good, None, flags=2)
        plt.imshow(img3)
        plt.show()






def Hamming(num1, num2):
    cont=0
    n1=num1
    n2=num2
    while n1>10:
        if n1 % 10 != n2 % 10:
            cont=cont+1
        n1=n1/10
        n2=n2/10
    if n1 != n2:
        cont=cont+1
    return cont







