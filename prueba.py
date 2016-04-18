import cv2

cap = cv2.VideoCapture(0)
multimediaVideo = cv2.VideoCapture("aumentado.avi")
frameActual = 0

while (cap.isOpened()):
    retVideo, multimediaVideoFrame = multimediaVideo.read()
    frameActual += 1
    # Para loop el video
    if frameActual == multimediaVideo.get(cv2.CAP_PROP_FRAME_COUNT):
        frameActual = 0
        multimediaVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cv2.imshow('video', multimediaVideoFrame)
    cv2.waitKey(100)

multimediaVideo.release()
cv2.destroyAllWindows()