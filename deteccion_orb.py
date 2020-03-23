import numpy as np
import cv2
from matplotlib import pyplot as plt
imagenes = []
kp = []
dc = []
keypoint = []
punto = []
sift = cv2.ORB_create(500, 1.3, 4, 31, 0, 2, 100, 31, 20)
#Cargo las fotos
for h in range(33):
    mejores = []
    nombre = "test/test"+str(h+1)+".jpg"
    img = cv2.imread(nombre, 0)
    imagenes.append(img)


    #Saco los keypoints y los descriptores de cada imagen y los almaceno
    kp1, des1 = sift.detectAndCompute(imagenes[h], None)
    kp.append(kp1)
    dc.append(des1)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    #Relleno el vector de votacion
    for i in kp:
        for j in i:
            keypoint.append(j)
            punto.append(j.pt)



    for i in range(48):
        img2 = cv2.imread("train/frontal_"+str(i+1)+".jpg", cv2.IMREAD_GRAYSCALE)  # trainImage
        kp2, descr2 = sift.detectAndCompute(img2, None)
        matches = bf.knnMatch(dc[h], descr2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])


        for p in good:
            indice1 = p[0].queryIdx
            keypoint1 = kp[h][indice1]
            mejores.append(keypoint1)


    keypts = sift.detect(imagenes[h])
    frame = cv2.drawKeypoints(imagenes[h], mejores, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)


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







