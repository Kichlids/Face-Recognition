import cv2
import numpy as np
import face_recognition

video = cv2.VideoCapture(0)

window_name = 'Face'


def filter_face(img):
    pass


# Read in Kichang face
kichang_img = face_recognition.load_image_file('Face_Images/Kichang.jpg')
kichang_img = cv2.cvtColor(kichang_img, cv2.COLOR_BGR2RGB)
kichang_img = cv2.rotate(kichang_img, cv2.ROTATE_90_CLOCKWISE)
#cv2.imshow('', kichang_img)

# Encode Kichang face to be compared
kichang_encode = face_recognition.face_encodings(kichang_img)[0]
#print(kichang_encode)

# Read in Jazmin face
jazmin_img = face_recognition.load_image_file('Face_Images/Jazmin.jpg')
jazmin_img = cv2.cvtColor(jazmin_img, cv2.COLOR_BGR2RGB)
#cv2.imshow('', jazmin_img)

while True:
    ret, frame = video.read()

    if ret: 

        # Get all coordinates with face
        face_locations = face_recognition.face_locations(frame)

        for face in face_locations:
            
            y1 = face[0] # top right
            x1= face[1] # top right
            y2 = face[2] # bottom left
            x2 = face[3] # bottom left

            bottom_left_point = (x2, y1)
            top_right_point = (x1, y2)
            
            # Positive difference between x's and y's
            dx = x1 - x2
            dy = y2 - y1
            
            # Extract the face from the raw image
            sample = frame[y1:y1+dy, x2:x2+dx]

            encode_samples = face_recognition.face_encodings(sample)

            # Proceed if a face snippet was extracted
            if len(encode_samples) > 0:
                encode_sample = encode_samples[0]
                # result = face_recognition.compare_faces([kichang_encode], sample)
                # print(result)

                face_distance = face_recognition.face_distance([kichang_encode], encode_sample)
                #print(face_distance)

                if face_distance < 0.5:
                    print(face_distance, 'face matched')
                    cv2.rectangle(frame, bottom_left_point, top_right_point, (255, 0, 0), 2)

                else:
                    print(face_distance, 'face not found')
                    cv2.rectangle(frame, bottom_left_point, top_right_point, (0, 0, 255), 2)




        cv2.imshow(window_name, frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

