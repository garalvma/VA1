import numpy as np
import cv2
from matplotlib import pyplot as plt
imagenes = []
kpTrain = []
dcTrain = []
kpTest = []
dcTest = []
keypoint = []
punto = []

sift = cv2.xfeatures2d.SIFT_create()

#Cargo las fotos TRAIN
for t in range(48):
    nombre = "train/frontal_" + str(t + 1) + ".jpg"
    img = cv2.imread(nombre, 0)

    # Saco los keypoints y los descriptores de cada imagen y los almaceno
    kp1, des1 = sift.detectAndCompute(img, None)
    kpTrain.append(kp1)
    dcTrain.append(des1)

    FLANN_INDEX_LSH = 5
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=3, multi_probe_level=1)
    search_params = dict(checks=-1)  # Número máximo de hojas a visitar cuando se busca vecinos
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    flann.add([des1])


#Cargo las fotos TEST
for h in range(33):
    mejores = []
    nombre = "test/test"+str(h+1)+".jpg"
    img = cv2.imread(nombre, 0)
    imagenes.append(img)

    kp2, des2 = sift.detectAndCompute(img, None)
    kpTest.append(kp2)
    dcTest.append(des2)
    matches=[]
    matches = flann.knnMatch(des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    for p in good:
        indice1 = p[0].queryIdx
        keypoint1 = kpTest[h][indice1]
        mejores.append(keypoint1)

    edges_img = cv2.Canny(img, 50, 150, apertureSize=3)
    # resolución de rho, theta y número de puntos mínimo para considerar recta
    lines = cv2.HoughLines(edges_img, 1, np.pi / 180, 100)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(color, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.figure(figsize=(15, 15))
plt.imshow(color)
plt.show()

    #frame = cv2.drawKeypoints(imagenes[h], mejores, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow('frame', frame)
    #cv2.waitKey(0)