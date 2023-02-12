# import the necessary packages
from keras.preprocessing.image import image_utils
from keras.models import load_model
import numpy as np
import imutils
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--cascade', required=True, help='path to where the face cascade resides')
ap.add_argument('-m', '--model', required=True, help='path to the pre-trained smile detector CNN')
ap.add_argument('-v', '--video', help='path to the (optional) video file')
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(args['cascade'])
model = load_model(args['model'])

# detect function
def detect(gray, frame):
    # detect faces in the input frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (fx, fy, fw, fh) in faces:
        # extract the ROI of the face from the grayscale image
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fy:fy + fh, fx:fx + fw]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = image_utils.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both 'smiling' and 'not smiling',
        # then set the label accordingly
        (not_smiling, smiling) = model.predict(roi)[0]
        label = 'Smiling' if smiling > not_smiling else "Not Smiling"

        # display the label and bounding box on the output frame
        color = (0, 0, 0)
        if label == 'Smiling':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        
        cv2.putText(frame, label, (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

    return frame

# grab the references to the webcam if a video path was not supplied
if not args.get('video', False):
    print('[INFO] starting video capture...')
    cap = cv2.VideoCapture(0)

# load the video otherwise
else:
    cap = cv2.VideoCapture(args['video'])

# grab width and height from video feed
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('videos/cnn.avi', cv2.VideoWriter_fourcc(*'XVID'), 12, (width, height))

# keep looping while the video capture is open
while cap.isOpened():
    # grab the current frame
    (grabbed, frame) = cap.read()

    # if we are viewing a video and we did no grab a frame, then we
    # have reached the end of the video
    if args.get('video') and not grabbed:
        break

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we draw on it later in the program
    copy = frame.copy()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # call the detect() function
    canvas = detect(gray, copy)

    writer.write(canvas)

    # show our detected face along with smiling/not smiling labels
    cv2.imshow('Face', copy)

    # stop the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clean up everything once all the processing is done
cap.release()
writer.release()
cv2.destroyAllWindows()
