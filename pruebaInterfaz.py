#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import subprocess as sp # Para ejecutar el otro script desde la línea de comandos
import winsound as ws # Para la reproducción de sonidos
from Tkinter import Tk # Python 2.7
from tkFileDialog import askopenfilename # Ventana de selección para el archivo multimedia
#from pruebaGrabacionSonido import grabarSonido
import interfaz as gui # Clase con la interfaz
import PyQt4.QtGui as qtgui # Para acceder a los métodos de los elementos de la interfaz desde este archivo

# Faltan por hacer (que recuerdo en este momento: 26/03/2016):
    # Implementar la opción de que el usuario pueda escoger el archivo multimedia (imagen, audio o video)        (✓)
    # Implementar la sobreposición de videos         (✓)
    # Implementar la sobreposición de sonidos         (✓)
    # Escalar el elemento multimedia (imagen o video) al tamaño del ROI       (✓ - Sólo video)
    # Manejar el error que se obtiene al intentar mover el elemento multimedia (imagen o video) mas allá de los límites del frame
    # Implementar controles para la reproducción del video
    # Implementar la opción de guardar o no el nuevo video (por ahora siempre lo graba)         (✓)
    # Grabar el nuevo video con audio si se escoje esta opción
    # Implementar una GUI         (✓)

# Variables 'globales' de control
frameActual = 0 # Controla el loop del video sobrepuesto y el video precargado
estaReproduciendo = False # Controla la reproduccion del sonido durante la detección
global umbral
umbral = 0 # Valor inicial. Entre 0 y 1 por haber usado "TM_CCOEFF_NORMED". Un porcentaje alto es mejor para ROIs pequeñas y viceversa
global root
root = Tk()
root.withdraw() # Para evitar que se abra la root window
global tipoSobreposicion
tipoSobreposicion = cv2.NORMAL_CLONE # Valor inicial

def definirUmbral(nivel):
    global umbral
    umbral = nivel/100  # Entre 0 y 1 por haber usado "TM_CCOEFF_NORMED". Un nivel alto es mejor para ROIs pequeñas y viceversa
                        # Se divide entre 100 porque el slider sólo devuelve valores enteros
    return umbral

def definirOpcionMultimedia(opcion):
    global opcionMultimedia
    opcionMultimedia = opcion  # 0 - Imagen, 1 - Video, 2 - Audio
    return opcionMultimedia

def escogerArchivo():
    # Para el título de la ventana de selección al escoger el archivo multimedia
    if opcionMultimedia == 0:
        opcion = 'Imagen'
    elif opcionMultimedia == 1:
        opcion = 'Video'
    else:
        opcion = 'Audio'

    archivo = askopenfilename(title='Escoja el archivo multimedia a sobreponer. DEBE SER UN ARCHIVO DE: ' + opcion)

    return archivo

def definirArchivoMultimedia(archivo):
    global multimedia
    global multimediaAudio
    global multimediaVideo
    # Determina cada archivo multimedia
    if opcionMultimedia == 0:
        multimedia = cv2.imread(archivo)
        return multimedia
    elif opcionMultimedia == 2:
        multimediaAudio = archivo
        return  multimediaAudio
    else:
        # El archivo seleccionado DEBE ser .avi (para mi PC en particular, a menos que se tengan instalados los códecs para otro formato)
        # Además, NO se puede escoger el mismo archivo generado 'aumentado.avi' si todavía está dentro de la carpeta con los archivos
        # porque el nuevo video que se va a crear y que haría uso del anterior ya creado, son el mismo!
        multimediaVideo = cv2.VideoCapture(archivo)
        return multimediaVideo

def definirGuardar(guardar):
    guardarVideoAumentado = guardar  # Si se guarda el video al final o no
    return guardarVideoAumentado

def definirFuente(camara):
    global usarCamara
    usarCamara = camara # Si usar un livestream o no
    # Para usar con un livestream o un video precargado
    if usarCamara == True:
        cap = cv2.VideoCapture(0)
    else:
        # DEBE ser un .avi por tener instalado el códec XVID. O cualquier otro formato si se tiene el códec adecuado
        videoPrecargado = askopenfilename(title='Escoja el video que se va a aumentar')
        cap = cv2.VideoCapture(videoPrecargado)
    return cap

def definirGrabacion(grabar):
    global guardarVideoAumentado
    guardarVideoAumentado = grabar
    global videoAumentado
    # Si se escoge la opción de guardar el video aumentado final, se inicializan las variables necesarias para almacenarlo
    if guardarVideoAumentado == True:
        fourcc = cv2.VideoWriter_fourcc(
            *'XVID')  # Requiere instalar el códec adecuado/escogido (en nuestro caso XVID o avi)
        # Sólo 4 FPS porque está capturando sólo a ese rate supongo que debido a la capacidad de mi PC
        videoAumentado = cv2.VideoWriter('aumentado.avi', fourcc, 4, (640, 480))
        return videoAumentado

def definirTipoSobreposicion(tipo):
    global tipoSobreposicion
    if tipo == 0:
        tipoSobreposicion = cv2.NORMAL_CLONE
    else:
        tipoSobreposicion = cv2.MIXED_CLONE
    return tipoSobreposicion

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
    sobrepuesto = cv2.seamlessClone(multimedia, frame, mascara, centroROI, tipoSobreposicion)
    cv2.imshow("Sobrepuesto", sobrepuesto)  # Muestra la sobreposición de la multimedia en la webcam/video
    # Captura los frames del video aumentado si la opción está habilitada
    if guardarVideoAumentado == True:
        videoAumentado.write(sobrepuesto)

def realizarProcesamiento():
    global frameActual
    global estaReproduciendo
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

        print(umbral)
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
            if opcionMultimedia == 2:
                ws.PlaySound(None, ws.SND_ASYNC | ws.SND_LOOP)  # Detiene la reproducción de audio
            break

# .connect() sólo acepta una función sin parámetros como argumento. Por lo tanto, se crean funciones sólo para llamar
# a las originales
def btnSiCamara():
    global cap
    cap = definirFuente(True)

def btnNoCamara():
    global cap
    cap = definirFuente(False)

def btnSiGrabar():
    definirGrabacion(True)

def btnNoGrabar():
    definirGrabacion(False)

def btnProcesar():
    realizarProcesamiento()

def btnArchivo():
    global archivo
    archivo = escogerArchivo()
    definirArchivoMultimedia(archivo)

def sliderUmbral():
    global umbral
    umbral = ui.sliderUmbral.value()/100.0

def btnImagen():
    definirOpcionMultimedia(0)

def btnVideo():
    definirOpcionMultimedia(1)

def btnAudio():
    definirOpcionMultimedia(2)

def btnNormalClone():
    definirTipoSobreposicion(0)

def btnMixedClone():
    definirTipoSobreposicion(1)

def cerrar():
    sys.exit(app.exec_())

# Ejecuta el código completo luego de definir las funciones, de manera directa desde el código
#realizarProcesamiento(estaReproduciendo, frameActual, cap)
#finalizar()

# Para crear la interfaz
if __name__ == "__main__":
    import sys

    # Crea las variables necesarias de QT
    app = qtgui.QApplication(sys.argv)
    MainWindow = qtgui.QMainWindow()
    # Instancia la clase de la interfaz
    ui = gui.Ui_MainWindow()
    ui.setupUi(MainWindow)

    # Une los elementos de la interfaz a las funciones definidas
    ui.btnSiCamara.toggled.connect(btnSiCamara)
    ui.btnNoCamara.toggled.connect(btnNoCamara)
    ui.btnSiGrabar.toggled.connect(btnSiGrabar)
    ui.btnNoGrabar.toggled.connect(btnNoGrabar)
    ui.sliderUmbral.valueChanged.connect(sliderUmbral)
    ui.btnImagen.toggled.connect(btnImagen)
    ui.btnVideo.toggled.connect(btnVideo)
    ui.btnAudio.toggled.connect(btnAudio)
    ui.btnNormalClone.toggled.connect(btnNormalClone)
    ui.btnMixedClone.toggled.connect(btnMixedClone)
    ui.btnSalir.clicked.connect(cerrar)
    ui.btnArchivo.clicked.connect(btnArchivo)
    ui.btnProcesar.clicked.connect(btnProcesar)

    MainWindow.show() # Muestra la ventana
    sys.exit(app.exec_()) # Evita que se cierre inmediatamente
