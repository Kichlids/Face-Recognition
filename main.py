import cv2
import numpy as np
import face_recognition

video = cv2.VideoCapture(0)

window_name = 'Face'


def filter_face(img):
    pass


while True:
    ret, frame = video.read()

    if ret: 

        # Get all coordinates with face
        face_locations = face_recognition.face_locations(frame)
        #print(face_locations)

        for face in face_locations:
            cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (255, 0, 255), 2)

        cv2.imshow(window_name, frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

