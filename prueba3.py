#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import subprocess as sp
import winsound as ws

# Faltan por hacer (que recuerdo en este momento: 26/03/2016):
    # Implementar la opción de que el usuario pueda escoger el archivo multimedia (imagen, audio o video)        (✓)
    # Implementar la sobreposición de videos
    # Implementar la sobreposición de sonidos         (✓)
    # Escalar el elemento multimedia (imagen o video) al tamaño del ROI       (✓)
    # Manejar el error que se obtiene al intentar mover el elemento multimedia (imagen o video) mas allá de los límites del frame
    # Implementar controles para la reproducción del video
    # Implementar la opción de guardar o no el nuevo video (por ahora siempre lo graba)
    # Implementar una GUI

opcionMultimedia = 2  # 0 - Imagen, 1 - Video, 2 - Audio

multimedia = cv2.imread("oni.png")  # El usuario debería poder seleccionar el archivo (y no ser sólo una imagen)

if opcionMultimedia == 0:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Requiere instalar el códec adecuado/escogido
    # Sólo 4 FPS porque está capturando sólo a ese rate supongo que debido a la capacidad de mi PC
    videoAumentado = cv2.VideoWriter('aumentado.avi', fourcc, 4, (640, 480))

multimediaAudio = "handel.wav"
estaReproduciendo = False

umbral = 0.8 # Entre 0 y 1 por haber usado "TM_CCOEFF_NORMED". Un porcentaje alto es mejor para ROIs pequeñas y viceversa

# Para usar con un video (no sé por qué no me funciona dando una ruta al archivo. Por ahora uso la webcam/cualquier camara)
cap = cv2.VideoCapture(0)
multimediaVideo = cv2.VideoCapture("aumentado.avi")
frameActual = 0

# Se ejecuta mientras el video/webcam se encuentre abierto
while (cap.isOpened()):
    ret, frame = cap.read()
    #multimediaVideoFrame = multimediaVideo.read()
    cv2.imshow('Presione la tecla "i" para seleccionar un ROI', frame)

    # El template debe obtenerse a partir de un frame del video original/webcam
    # Determina si el usuario presiona una tecla
    key = cv2.waitKey(1) & 0xFF
    # Si presiona la tecla "i"
    if key == ord("i"):
        # Captura el frame actual para usarse como el template
        cv2.imwrite("frame.jpg", frame)
        sp.call(["python", "click_and_crop.py", "--image", "frame.jpg"])  # Ejecuta el otro script para obtener la ROI

    # Realiza el procesamiento de Template matching
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

        if opcionMultimedia == 0:
            # Para hallar el centro del rectángulo del ROI
            centroROI = (bottom_right[0] - (bottom_right[0] / 4), bottom_right[1] - (bottom_right[1] / 4))
            # Para la máscara (Es del tamaño del elemento multimedia, e inicialmente color negra)
            mascara = np.zeros(multimedia.shape, multimedia.dtype)
            # Arreglo de puntos que determina un polígono (cuadrado) del tamaño del frame. Usado para determinar la zona visible (blanca) la máscara
            poly = np.array([[0, 0], [0, 400], [frame.shape[0], frame.shape[1]], [frame.shape[1], 0]], np.int32)
            # Pinta la máscara visible (blanco) a partir del contorno generado por los polígonos anteriormente definidos
            cv2.fillPoly(mascara, [poly], (255, 255, 255))
            #multimedia = cv2.resize(multimedia, (w*2, h*2))
            # Genera la sobreposición de la multimedia y la muestra en el centro del ROI
            sobrepuesto = cv2.seamlessClone(multimedia, frame, mascara, centroROI, cv2.MIXED_CLONE)
            cv2.imshow("Sobrepuesto", sobrepuesto) # Muestra la sobreposición de la multimedia en la webcam/video
            # Captura los frames del video aumentado si presiona la tecla 'r'
            videoAumentado.write(sobrepuesto)

            # OTROS INTENTOS (FALLIDOS) PARA IMPLEMENTAR LA SOBREPOSICIÓN

            #frame[0:multimedia.shape[0], 0:multimedia.shape[1]] = multimedia
            #cv2.imshow("imagen", frame)

            #alpha = 1
            #beta = 0
            #gamma = 0
            #overlay = frame.copy()
            #overlay = cv2.resize(multimedia, (frame.shape[1], frame.shape[0]))
            #overlay = cv2.resize(multimedia, (w,h))
            #cv2.imshow("overlay", overlay)
            #overlay = cv2.resize(multimedia, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
            #overlay = cv2.resize(multimedia, ())
            #cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 255), -1)
            #frame = cv2.addWeighted(overlay, alpha, template, beta, gamma)
            #cv2.imshow("Sobrepuesto", frame)
            #videoAumentado.write(frame)

            #print (top_left, bottom_right)

            #np.copyto(frame, multimedia)
            #cv2.imshow("Sobrepuesto", frame)
            #videoAumentado.write(frame)

        elif opcionMultimedia == 1:
            retVideo, multimediaVideoFrame = multimediaVideo.read()

            # Stretch a la forma del ROI
            multimediaVideoFrame = cv2.resize(multimediaVideoFrame, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
            frameActual += 1
            # Para loop el video
            if frameActual == multimediaVideo.get(cv2.CAP_PROP_FRAME_COUNT):
                frameActual = 0
                multimediaVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cv2.waitKey(1)
            # Para hallar el centro del rectángulo del ROI
            centroROI = (bottom_right[0] - (bottom_right[0] / 4), bottom_right[1] - (bottom_right[1] / 4))
            # Para la máscara (Es del tamaño del elemento multimedia, e inicialmente color negra)
            mascara = np.zeros(multimediaVideoFrame.shape, multimediaVideoFrame.dtype)
            # Arreglo de puntos que determina un polígono (cuadrado) del tamaño del frame. Usado para determinar la zona visible (blanca) la máscara
            poly = np.array([[0, 0], [0, 400], [frame.shape[0], frame.shape[1]], [frame.shape[1], 0]], np.int32)
            # Pinta la máscara visible (blanco) a partir del contorno generado por los polígonos anteriormente definidos
            cv2.fillPoly(mascara, [poly], (255, 255, 255))
            # Genera la sobreposición de la multimedia y la muestra en el centro del ROI
            sobrepuesto = cv2.seamlessClone(multimediaVideoFrame, frame, mascara, centroROI, cv2.MIXED_CLONE)
            cv2.imshow("Sobrepuesto", sobrepuesto)  # Muestra la sobreposición de la multimedia en la webcam/video
            # Captura los frames del video aumentado si presiona la tecla 'r'
            #videoAumentado.write(sobrepuesto)
        elif opcionMultimedia == 2:
            if estaReproduciendo == False:
                ws.PlaySound(multimediaAudio, ws.SND_ASYNC | ws.SND_LOOP)
                estaReproduciendo = True
    else:
        if opcionMultimedia == 0 or opcionMultimedia == 1:
            cv2.destroyWindow("Sobrepuesto") # Cierra la ventana con la sobreposición de multimedia
        elif opcionMultimedia == 2:
            estaReproduciendo = False
            ws.PlaySound(None, ws.SND_ASYNC | ws.SND_LOOP)  # Para parar la reproducción de audio

    #cv2.imshow("res", res) # Muestra el mapa del "parecido" de cada píxel con el template (Operación de vecindad)
    cv2.imshow("img", img) # Muestra la detección y seguimiento del template en la webcam/video

    # Termina el loop presionando 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if opcionMultimedia == 0:
    videoAumentado.release()
cv2.destroyAllWindows()