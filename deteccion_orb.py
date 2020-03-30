import numpy as np
import cv2
import math

def maximo(g, m, n):
    max = 0
    b = 0
    v = 0
    for i in range(m):
        for j in range(n):
            if g[i][j] > max:
                b=i
                v=j
                max=g[i][j]
    return (b,v)

keypoints = []
centro = []
angulo = []
descriptor = []
orientacion = []
escala = []
keypoints2 = []
keypoints3 = []
descriptor2=[]

sift = sift = cv2.ORB_create(100, 1.3, 4, 31, 0, 2, cv2.ORB_HARRIS_SCORE, 31, 20)

#Imagenes de train
for i in range(48):
    nombre = "train/frontal_" + str(i + 1) + ".jpg"
    img = cv2.imread(nombre, 0)

    kp1, des1 = sift.detectAndCompute(img, None)

    width2, height2 = img.shape[:2]
    cen = (int(width2/2), int(height2/2))

    #Almaceno todos los datos de todas las imagenes para crear el entrenamiento
    longitud1 = len(kp1)
    for j in range(longitud1):
        keypoints.append(kp1[j])
        descriptor.append(des1[j])
        orientacion.append(kp1[j].angle)
        escala.append(kp1[j].size)
        restax = kp1[j].pt[0] - cen[0]
        restay = kp1[j].pt[1] - cen[1]
        n1 = math.pow(restax, 2)
        n2 = math.pow(restay, 2)
        num = math.sqrt(n1+n2)
        centro.append(num)
        res = restay / num
        ang=math.asin(res)
        resul = math.degrees(ang)
        angulo.append(resul)

bf = cv2.BFMatcher()

#Trabajo sobre las imagenes de test
for k in range(33):
    nombre = "test/test" + str(k + 1) + ".jpg"
    img = cv2.imread(nombre, 0)

    keypoints3 = []
    width, height = img.shape[:2]
    a = np.zeros((int(width), int(height)))

    kp2, des2 = sift.detectAndCompute(img, None)

    longitud2 = len(kp2)
    for l in range(longitud2):
        keypoints2.append(kp2[l])
        descriptor2.append(des2[l])
    matches = bf.knnMatch(np.asarray(des2,), np.asarray(descriptor,), k=3)

    #Relleno el vector de acumulacion a en las coordenadas que aparecen en el entrenamiento /10 para que no se salgan los puntos (ya que el vector tambien esta /10)
    for m in matches:
        for n in range(len(m)):
            pos2 = m[n].queryIdx
            #Posicion training
            pos1 = m[n].trainIdx
            keypoints3.append(keypoints[pos1])
            kp=keypoints[pos1]
            escala2 = int(escala[pos2])
            escala1 = int(escala[pos1])
            anguloTrain = angulo[pos1]
            #Saco el punto del keypoint del entrenamiento
            x = keypoints[pos1].pt[0]
            y = keypoints[pos1].pt[1]
            x = x+(escala2/escala1)*centro[pos1]
            y = y+(escala2/escala1)*centro[pos1]

            #Saco el angulo correspondiente al modulo del vector que va del KP al centro
            an=(kp.angle+anguloTrain+keypoints2[pos2].angle)
            if an > 180:
                an = (an -180)*(-1)
            elif an < -180:
                an = (an +180)*(-1)

            keypoints3[pos2].angle = an
            keypoints3[pos2].pt = (int(x), int(y))
            #Añado 1 al vector de votacion
            if int(x)<int(width) and int(y)<int(height):
                a[int(x)][int(y)] = a[int(x)][int(y)]+1

    #Saco del vector de votacion el numero mas alto y lo pinto como un cuadrado en la imagen de test
    for z in range(1):
        max1 = maximo(a, int(width), int(height))
        a[max1[0]][max1[1]]=0
        frame = cv2.drawMarker(img, max1, (0, 0, 0), cv2.MARKER_SQUARE, 20, 2, cv2.FILLED)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)