'''
ECE 4973 Computer Vision
Spring 2021
University of Oklahoma
Dr. Samuel Cheng

Authors:
    Jazmin Gomez
    Kichang Song
'''

import cv2
import face_recognition

video = cv2.VideoCapture(0)

window_name = 'Facial Recognition Project'



def anonymize_face_simple(image, factor=3.0):
    (h,w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)

    if kW % 2 == 0:
        kW -= 1

    if kH % 2 == 0:
        kH -= 1

    return cv2.GaussianBlur(image, (kW, kH), 0)

# Read in Kichang face
kichang_img = face_recognition.load_image_file('Face_Images/Kichang.jpg')
kichang_img = cv2.cvtColor(kichang_img, cv2.COLOR_BGR2RGB)
kichang_img = cv2.rotate(kichang_img, cv2.ROTATE_90_CLOCKWISE)

# Encode Kichang face to be compared
kichang_encode = face_recognition.face_encodings(kichang_img)[0]

# Read in Jazmin face
jazmin_img = face_recognition.load_image_file('Face_Images/Jazmin.jpg')
jazmin_img = cv2.cvtColor(jazmin_img, cv2.COLOR_BGR2RGB)

# Encode Jazmin face to be compared
jazmin_encode = face_recognition.face_encodings(jazmin_img)[0]


# Determine which face will be filtered
encode_filtered = jazmin_encode




# Threshold for determining faces
# Must be between 0 and 1, where closer to 0 indicates higher probability that the faces matches
FACE_DISTANCE_CONSTANT = 0.5

while True:

    # Read from main camera
    ret, frame = video.read()

    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    if ret: 
        # Find all face locations in the frame
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

            # Encode the face snippet to be compared
            encode_samples = face_recognition.face_encodings(sample)

            # Proceed if a face snippet was extracted
            if len(encode_samples) > 0:
                # Get the first (and only) element
                encode_sample = encode_samples[0]
                
                # Calculate a value between 0 and 1 for comparing two faces
                # Closer to 0 indicates higher probability that the two faces match
                face_distance = face_recognition.face_distance([encode_filtered], encode_sample)

                if face_distance < FACE_DISTANCE_CONSTANT:
                    print(face_distance, 'face matched')

                    # Put a blue rectangle if face that needs to be filtered is found
                    cv2.rectangle(frame, bottom_left_point, top_right_point, (255, 0, 0), 2)
                    blur_img = anonymize_face_simple(sample)
                    cv2.imshow('Blurred', blur_img)

                    # Apply the filter on the face
                    frame[y1:y1+dy, x2:x2+dx] = blur_img
                else:
                    print(face_distance, 'face not found')
                    
                    # Put a red rectangle if face that needs to be filtered is not found
                    cv2.rectangle(frame, bottom_left_point, top_right_point, (0, 0, 255), 2)
        
        # Show the final output
        cv2.imshow(window_name, frame)

    # Exit program on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

