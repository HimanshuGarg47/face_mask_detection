import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model = load_model("mask_recog.h5")

def face_mask_detector(frame):
  # frame = cv2.imr ead(fileName)

  # Converts an image from one color space to another. The function converts an input image from one color space to another
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # color space conversion code

  # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles.
  faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
  faces_list=[]
  preds=[]
  for (x, y, w, h) in faces:
        # frame[y:y+h, x:x+w] specifies a slice of the frame array, where y and x are the starting indices of the row and column ranges, respectively, and h and w are the heights and widths of the sub-array to be extracted.-
      face_frame = frame[y:y+h,x:x+w]
      face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
      face_frame = cv2.resize(face_frame, (224, 224))
      face_frame = img_to_array(face_frame)

      # Expand the shape of an array. Insert a new axis that will appear at the axis position in the expanded array shape.
      face_frame = np.expand_dims(face_frame, axis=0)
      face_frame =  preprocess_input(face_frame)
      faces_list.append(face_frame
      )
      if len(faces_list)>0:
        """Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for batch processing of large numbers of inputs. It is not intended for use inside of loops that iterate over your data and process small numbers of inputs at a time.
        """
        preds = model.predict(faces_list) # Return Numpy array(s) of predictions.

      for pred in preds:
          (mask, withoutMask) = pred

      label = "Mask" if mask > withoutMask else "No Mask"
      color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
      cv2.putText(frame, label, (x, y- 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

      cv2.rectangle(frame, (x, y), (x + w, y + h),color, 3)
  # cv2_imshow(frame)
  return frame




"""To capture a video, you need to create a VideoCapture object. Its argument can be either the device index or the name of a video file. A device index is just the number to specify which camera. Normally one camera will be connected (as in my case). So I simply pass 0 (or -1). You can select the second camera by passing 1 and so on. After that, you can capture frame-by-frame
    """
cap = cv2.VideoCapture(0)

# cap.read() returns a bool (True/False). If the frame is read correctly, it will be True. So you can check for the end of the video by checking this returned value.
# ret, frame = cap.read()
# frame_height, frame_width, _ = frame.shape
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
print("Processing Video...")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while cap.isOpened(): # Returns true if video writer has been successfully initialized.
  ret, frame = cap.read()
  if not ret:
    out.release() # Closes the video writer.
    break
  output = face_mask_detector(frame)
  out.write(output) # Writes the next video frame.

# Closes the video writer.
out.release()
print("Done processing video")