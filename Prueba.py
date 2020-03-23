import numpy as np
import cv2
from matplotlib import pyplot as plt

imagenes = []
kp = []
dc = []
keypoint = []
punto = []

#Cargo las fotos
for i in range(48):
    nombre = "train/frontal_"+str(i+1)+".jpg"
    img = cv2.imread(nombre, 0)
    imagenes.append(img)

sift = cv2.xfeatures2d.SIFT_create()

#Saco los keypoints y los descriptores de cada imagen y los almaceno
for i in range(48):
    kp1, des1 = sift.detectAndCompute(imagenes[i], None)
    kp.append(kp1)
    dc.append(des1)

#Relleno el vector de votacion
for i in kp:
    for j in i:
        keypoint.append(j)
        punto.append(j.pt)



FLANN_INDEX_LSH = 6
index_params= dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=3,multi_probe_level=1)
search_params = dict(checks=-1) # Número máximo de hojas a visitar cuando se busca vecinos
flann = cv2.FlannBasedMatcher(index_params,search_params)

for d in dc:
    flann.add([d])

results = flann.knnMatch(np.array([[8,8,8]],dtype=np.uint8),k=3)
flann.kn



for r in results:
  for m in r:
    print("Dist:",m.distance," img:",m.imgIdx," queryIdx:", m.queryIdx," trainIdx:",m.trainIdx)
