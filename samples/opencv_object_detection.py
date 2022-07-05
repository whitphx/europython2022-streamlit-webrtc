from typing import NamedTuple, List
from pathlib import Path

import cv2
import numpy as np


CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class Detection(NamedTuple):
    name: str
    prob: float


confidence_threshold = 0.5


def _annotate_image(image, detections):
    # loop over the detections
    (h, w) = image.shape[:2]
    result: List[Detection] = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            name = CLASSES[idx]
            result.append(Detection(name=name, prob=float(confidence)))

            # display the prediction
            label = f"{name}: {round(confidence * 100, 2)}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLORS[idx],
                2,
            )
    return image, result


HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent

MODEL_LOCAL_PATH = ROOT.parent / "streamlit-webrtc/models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_LOCAL_PATH = ROOT.parent / "streamlit-webrtc/models/MobileNetSSD_deploy.prototxt.txt"


net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))


def AWESOME_IMAGE_FILTER(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    detections = net.forward()
    annotated_image, result = _annotate_image(image, detections)
    return annotated_image


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    output = AWESOME_IMAGE_FILTER(frame)

    cv2.imshow('frame', output)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Ref: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
