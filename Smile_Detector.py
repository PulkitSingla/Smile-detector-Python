import cv2

# Face classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # read the current frame from the webcam video stream
    successful_frame_read, frame = webcam.read()

# if there is an error, abort
    if not successful_frame_read:
        break
# Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)


# Run smile detection within each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        the_face = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=5)

        # for (x_, y_, w_, h_) in smiles:
        # cv2.rectangle(the_face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
        else:
            cv2.putText(frame, 'Not Smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
# show the current frame
    cv2.imshow('Smile Detector', frame)
# display
    cv2.waitKey(1)
# cleanup
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")
