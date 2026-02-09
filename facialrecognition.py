import face_recognition
import cv2
import os
import numpy as np

# Step 1: Load known images and encode them
known_face_encodings = []
known_face_names = []

dataset_dir = '/dataset/Leonel'

for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        
        known_face_encodings.append(face_encoding)
        known_face_names.append(person_name)

# Step 2: Initialize the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Step 3: Capture frame-by-frame
    ret, frame = video_capture.read()

    # Step 4: Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Step 5: Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Step 6: Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for any known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Step 7: Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Step 8: Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Step 9: Display the resulting image
    cv2.imshow('Video', frame)

    # Step 10: Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 11: Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
