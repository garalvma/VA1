import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('test/test9.jpg',0)
edges_img = cv2.Canny(img,50,150,apertureSize = 3)
#resolución de rho, theta y número de puntos mínimo para considerar recta
lines = cv2.HoughLines(edges_img,1,np.pi/180,180)
color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
for line in lines:
 rho,theta = line[0]
 a = np.cos(theta)
 b = np.sin(theta)
 x0 = a*rho
 y0 = b*rho
 x1 = int(x0 + 1000*(-b))
 y1 = int(y0 + 1000*(a))
 x2 = int(x0 - 1000*(-b))
 y2 = int(y0 - 1000*(a))

 cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)
plt.figure(figsize=(15,15))
plt.imshow(color)
plt.show()