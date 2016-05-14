#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import subprocess as sp
import winsound as ws
from Tkinter import Tk # Python 2.7
from tkFileDialog import askopenfilename
#from pruebaGrabacionSonido import grabarSonido

# Faltan por hacer (que recuerdo en este momento: 26/03/2016):
    # Implementar la opción de que el usuario pueda escoger el archivo multimedia (imagen, audio o video)        (✓)
    # Implementar la sobreposición de videos         (✓)
    # Implementar la sobreposición de sonidos         (✓)
    # Escalar el elemento multimedia (imagen o video) al tamaño del ROI       (✓ - Sólo video)
    # Manejar el error que se obtiene al intentar mover el elemento multimedia (imagen o video) mas allá de los límites del frame
    # Implementar controles para la reproducción del video
    # Implementar la opción de guardar o no el nuevo video (por ahora siempre lo graba)         (✓)
    # Grabar el nuevo video con audio si se escoje esta opción
    # Implementar una GUI

# Opciones iniciales
opcionMultimedia = 2  # 0 - Imagen, 1 - Video, 2 - Audio
umbral = 0.6 # Entre 0 y 1 por haber usado "TM_CCOEFF_NORMED". Un porcentaje alto es mejor para ROIs pequeñas y viceversa
guardarVideoAumentado = True # Si se guarda el video al final o no
usarCamara = False # Si usar un livestream o no

# Para abrir una ventana de diálogo y escojer los archivos deseados
root = Tk()
root.withdraw()

# Para el título del cuadro de diálogo al escoger el archivo Multimedia
if opcionMultimedia == 0:
    opcion = 'Imagen'
elif opcionMultimedia == 1:
    opcion = 'Video'
else:
    opcion = 'Audio'

archivo = askopenfilename(title='Escoja el archivo multimedia a sobreponer. DEBE SER UN ARCHIVO DE: ' + opcion)

if usarCamara == False:
    # DEBE ser un .avi por tener instalado el códec XVID. O cualquier otro formato si se tiene el códec adecuado
    videoPrecargado = askopenfilename(title='Escoja el video que se va a aumentar')

# Determina cada archivo multimedia
if opcionMultimedia == 0:
    multimedia = cv2.imread(archivo)
elif opcionMultimedia == 2:
    multimediaAudio = archivo
else:
    # El archivo seleccionado DEBE ser .avi (para mi PC en particular, a menos que se tengan instalados los códecs para otro formato)
    # Además, NO se puede escoger el mismo archivo generado 'aumentado.avi' si todavía está dentro de la carpeta con los archivos
    # porque el nuevo video que se va a crear y que haría uso del anterior ya creado, son el mismo!
    multimediaVideo = cv2.VideoCapture(archivo)

# Variables 'globales' de control
frameActual = 0 # Controla el loop del video sobrepuesto y el video precargado
estaReproduciendo = False # Controla la reproduccion del sonido

# Si se escoge la opción de guardar el video aumentado final, se inicializan las variables necesarias para almacenarlo
if guardarVideoAumentado == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Requiere instalar el códec adecuado/escogido (en nuestro caso XVID o avi)
    # Sólo 4 FPS porque está capturando sólo a ese rate supongo que debido a la capacidad de mi PC
    videoAumentado = cv2.VideoWriter('aumentado.avi', fourcc, 4, (640, 480))

# Para usar con un livestream o un video precargado
if usarCamara == True:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(videoPrecargado)

def obtenerROI(frame):
    cv2.imshow('Presione la tecla "i" para seleccionar una ROI', frame)

    # El template debe obtenerse a partir de un frame del video original/webcam
    # Determina si el usuario presiona una tecla
    key = cv2.waitKey(1) & 0xFF
    # Si presiona la tecla "i"
    if key == ord("i"):
        # Captura el frame actual para usarse como el template
        cv2.imwrite("frame.jpg", frame)
        sp.call(["python", "click_and_crop.py", "--image", "frame.jpg"])  # Ejecuta el otro script para obtener la ROI

def finalizar():
    cap.release()  # Libera la webcam/video
    if guardarVideoAumentado == True:
        videoAumentado.release()  # Libera el nuevo video aumentado
        # Para unir el video aumentado con el audio en caso de haber seleccionado esta opción
        # if opcionMultimedia == 2:
        # sp.call(["ffmpeg", "-i", "aumentado.avi", "-i", "grabacion.wav", "-vcodec", "copy", "-acodec", "copy",
        # "aumentadoAudio.avi"])  # Ejecuta ffmpeg para crear el nuevo video con audio
    cv2.destroyAllWindows()  # Cierra todas las ventanas

def posicionarMultimedia(frame, multimedia, esquinaCaja):
    # Para hallar el centro del rectángulo del ROI
    centroROI = (esquinaCaja[0] - (esquinaCaja[0] / 4), esquinaCaja[1] - (esquinaCaja[1] / 4))
    # Para la máscara (Es del tamaño del elemento multimedia, e inicialmente color negra)
    mascara = np.zeros(multimedia.shape, multimedia.dtype)
    # Arreglo de puntos que determina un polígono (cuadrado) del tamaño del frame. Usado para determinar la zona visible (blanca) la máscara
    poly = np.array([[0, 0], [0, 400], [frame.shape[0], frame.shape[1]], [frame.shape[1], 0]], np.int32)
    # Pinta la máscara visible (blanco) a partir del contorno generado por los polígonos anteriormente definidos
    cv2.fillPoly(mascara, [poly], (255, 255, 255))
    # Genera la sobreposición de la multimedia y la muestra en el centro del ROI
    sobrepuesto = cv2.seamlessClone(multimedia, frame, mascara, centroROI, cv2.NORMAL_CLONE)
    cv2.imshow("Sobrepuesto", sobrepuesto)  # Muestra la sobreposición de la multimedia en la webcam/video
    # Captura los frames del video aumentado si la opción está habilitada
    if guardarVideoAumentado == True:
        videoAumentado.write(sobrepuesto)

def realizarProcesamiento(estaReproduciendo, frameActual, video):
    # Se ejecuta mientras el video/webcam se encuentre abierto
    while (cap.isOpened()):

        # Termina el programa si no se seleccionó ningún archivo
        if archivo == '':
            break

        # Lee el video del livestream o del video precargado
        ret, frame = cap.read()

        # Si no está usando el livestream de una cámara, se debe loop el video precargado
        if usarCamara == False:
            frameActual += 1
            # Para loop el video precargado
            if frameActual == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                frameActual = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cv2.waitKey(1)

        obtenerROI(frame)

        # Realiza el procesamiento de Template matching
        frameGrayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Template matching sólo funciona con imágenes en escala de grises
        img = frameGrayscale
        template = cv2.imread('template.jpg', 0)
        w, h = template.shape[::-1]

        # Aplica Template matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        #if opcionMultimedia == 2:
            #grabarSonido()

        # Compara el valor con mayor aproximación a la imagen usada como template, con el umbral, por cada frame
        if max_val >= umbral:
            cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2) # Dibuja el rectángulo blanco

            if opcionMultimedia == 0:
                posicionarMultimedia(frame, multimedia, bottom_right)
            elif opcionMultimedia == 1:
                retVideo, multimediaVideoFrame = multimediaVideo.read()

                # Stretch el video a la forma del ROI
                multimediaVideoFrame = cv2.resize(multimediaVideoFrame, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
                frameActual += 1
                # Para loop el video sobrepuesto
                if frameActual == multimediaVideo.get(cv2.CAP_PROP_FRAME_COUNT):
                    frameActual = 0
                    multimediaVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cv2.waitKey(1)
                posicionarMultimedia(frame, multimediaVideoFrame, bottom_right)
            elif opcionMultimedia == 2:
                if estaReproduciendo == False:
                    ws.PlaySound(multimediaAudio, ws.SND_ASYNC | ws.SND_LOOP)
                    estaReproduciendo = True
                # Captura los frames del video aumentado si la opción está habilitada. Aún no captura el audio
                if guardarVideoAumentado == True:
                    videoAumentado.write(frame)
        else:
            if opcionMultimedia == 0 or opcionMultimedia == 1:
                cv2.destroyWindow("Sobrepuesto") # Cierra la ventana con la sobreposición de multimedia
            elif opcionMultimedia == 2:
                estaReproduciendo = False
                ws.PlaySound(None, ws.SND_ASYNC | ws.SND_LOOP)  # Detiene la reproducción de audio

        #cv2.imshow("res", res) # Muestra el mapa del "parecido" de cada píxel con el template (Operación de vecindad)
        cv2.imshow("img", img) # Muestra la detección y seguimiento del template en la webcam/video

        # Termina el loop presionando 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Ejecuta el código completo luego de definir las funciones
realizarProcesamiento(estaReproduciendo, frameActual, cap)
finalizar()