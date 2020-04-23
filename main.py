import FaceRecognize

camID = 0 # Initial camera ID of confirmed sighting
trail = []

def NextCam():
    pass

while True:
    flag = FaceRecognize.recognize(camID)

    if flag == 1:
        trail.append(camID)

    camID = NextCam()