# import the necessary packages
import cv2

# grab cascade classifiers
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade/haarcascade_smile.xml')

# detect function
def detect(gray, frame):
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (fx, fy, fw, fh) in faces:
        # draw rectangle for face
        cv2.rectangle(frame, (fx, fy), ((fx + fw), (fy + fh)), (255, 0, 0), 2)
        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = frame[fy:fy + fh, fx:fx + fw]
        # detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            # draw rectangle for smile
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)

    return frame

# grab the references to the webcam
print('[INFO] starting video capture...')
cap = cv2.VideoCapture(0)

# grab width and height from video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('videos/haar.avi', cv2.VideoWriter_fourcc(*'XVID'), 12, (width, height))

# keep looping while the video capture is open
while cap.isOpened():
    # grab the current frame
    _, frame = cap.read()

    # capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calls the detect() function
    canvas = detect(gray, frame)

    writer.write(canvas)

    # display the result on camera feed
    cv2.imshow('Video', canvas)

    # stop the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# clean up everything once all the processing is done
cap.release()
writer.release()
cv2.destroyAllWindows()
