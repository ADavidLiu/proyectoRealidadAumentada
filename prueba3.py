#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import subprocess as sp

# Faltan por hacer (que recuerdo en este momento: 26/03/2016):
    # Implementar la opción de que el usuario pueda escoger el archivo multimedia (imagen, audio o video)
    # Implementar la sobreposición de videos y sonidos (no sólo imágenes como ahora)
    # Escalar el elemento multimedia al tamaño del ROI
    # Manejar el error que se obtiene al intentar mover el elemento multimedia mas allá de los límites del frame
    # Implementar controles para la reproducción del video (pausar y reanudar)
    # Implementar la opción de guardar o no el nuevo video (por ahora siempre lo graba)
    # Tal vez implementar una GUI

multimedia = cv2.imread("oni.png")  # El usuario debería poder seleccionar el archivo (y no ser sólo una imagen)
#multimedia = cv2.VideoCapture("video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID') # Requiere instalar el códec adecuado/escogido
videoAumentado = cv2.VideoWriter('aumentado.avi',fourcc, 4, (640,480)) # Sólo 4 FPS porque está capturando sólo a ese rate
                                                                       # supongo que debido a la capacidad de mi PC

umbral = 0.2 # Entre 0 y 1 por haber usado "TM_CCOEFF_NORMED". Un porcentaje alto es mejor para ROIs pequeñas y viceversa

# Para usar con un video (no sé por qué no me funciona dando una ruta al archivo. Por ahora uso la webcam/cualquier camara)
cap = cv2.VideoCapture(0)

# Se ejecuta mientras el video/webcam se encuentre abierto
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('Presione la tecla "i" para seleccionar un ROI', frame)

    # El template debe obtenerse a partir de un frame del video original/webcam
    # Determina si el usuario presiona una tecla
    key = cv2.waitKey(1) & 0xFF
    # Si presiona la tecla "i"
    if key == ord("i"):
        # Captura el frame actual para usarse como el template
        cv2.imwrite("frame.jpg", frame)
        sp.call(["python", "click_and_crop.py", "--image", "frame.jpg"])  # Ejecuta el otro script para obtener la ROI

    # Realiza el procesamiento de Template matching (Código fuente del tutorial)
    frameGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Template matching sólo funciona con imágenes en escala de grises
    img = frameGrayscale
    img2 = img.copy()
    template = cv2.imread('template.jpg', 0)
    w, h = template.shape[::-1]

    img = img2.copy()

    # Aplica Template matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Compara el valor con mayor aproximación a la imagen usada como template, con el umbral, por cada frame
    if max_val >= umbral:
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)
        print "En rango"
        # Para hallar el centro del rectángulo del ROI
        centroROI = (bottom_right[0] - (bottom_right[0]/4), bottom_right[1] - (bottom_right[1]/4))
        # Para la máscara (Es del tamaño del elemento multimedia, e inicialmente color negra)
        mascara = np.zeros(multimedia.shape, multimedia.dtype)
        # Arreglo de puntos que determina un polígono (cuadrado). Usado para determinar la zona visible (blanca) la máscara
        poly = np.array([ [0,0], [0, 480], [640, 480], [640, 0] ], np.int32)
        # Pinta la máscara visible (blanco) a partir del contorno generado por los polígonos anteriormente definidos
        cv2.fillPoly(mascara, [poly], (255, 255, 255))
        # Genera la sobreposición de la multimedia y la muestra en el centro del ROI
        sobrepuesto = cv2.seamlessClone(multimedia, frame, mascara, centroROI, cv2.MIXED_CLONE)
        cv2.imshow("Sobrepuesto", sobrepuesto) # Muestra la sobreposición de la multimedia en la webcam/video
        # Captura los frames del video aumentado si presiona la tecla 'r'
        videoAumentado.write(sobrepuesto)
    else:
        print "Fuera de rango"
        cv2.destroyWindow("Sobrepuesto") # Cierra la ventana con la sobreposición de multimedia

    #cv2.imshow("res", res) # Muestra el mapa del "parecido" de cada píxel con el template
    cv2.imshow("img", img) # Muestra la detección y seguimiento del template en la webcam/video

    # Termina el loop presionando 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
videoAumentado.release()
cv2.destroyAllWindows()