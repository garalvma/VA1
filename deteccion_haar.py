import cv2
import argparse

def deteccion(frame):
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imagen2 = cv2.equalizeHist(imagen)
    puntos = coches.detectMultiScale(imagen2)

    for (x, y, x2, y2) in puntos:
        frame = cv2.rectangle(frame, (x, y), (x2, y2), color=cv2.COLOR_BGR2HSV, thickness=2, shift=0)

    cv2.imshow("frame", frame)

parser = argparse.ArgumentParser(description='Deteccion de coches.')
parser.add_argument('--coches', help='Ruta a coches', default='haar/coches.xml')
args = parser.parse_args()

coches_nombre = args.coches
coches = cv2.CascadeClassifier()

encontrado = cv2.samples.findFile(coches_nombre)
coches.load(encontrado)

for i in range(33):
    nombre = "train/frontal_" + str(i + 1) + ".jpg"
    img = cv2.imread(nombre)
    deteccion(img)
    cv2.waitKey(0)