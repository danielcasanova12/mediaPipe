# import cv2
# import mediapipe as mp
# import numpy as np
# cap = cv2.VideoCapture(0)
# mesh = mp.solutions.face_mesh
# drawing = mp.solutions.drawing_utils
#
#
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
#
# def draw_landmarks_on_image(rgb_image, detection_result):
#   pose_landmarks_list = detection_result.pose_landmarks
#   annotated_image = np.copy(rgb_image)
#
#   # Loop through the detected poses to visualize.
#   for idx in range(len(pose_landmarks_list)):
#     pose_landmarks = pose_landmarks_list[idx]
#
#     # Draw the pose landmarks.
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     pose_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       pose_landmarks_proto,
#       solutions.pose.POSE_CONNECTIONS,
#       solutions.drawing_styles.get_default_pose_landmarks_style())
#   return annotated_image
#
# # STEP 1: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import cv2
#
# # STEP 2: Create an PoseLandmarker object.
# base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
# options = vision.PoseLandmarkerOptions(
#     base_options=base_options,
#     output_segmentation_masks=True)
# detector = vision.PoseLandmarker.create_from_options(options)
#
# # STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")
#
# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)
#
# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
#
# segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
# cv2.imshow(visualized_mask)

# Importações necessárias
import cv2
import mediapipe as mp
import numpy as np
import torch
import time

def foundPeople(img, detectPose=False):
    result = yolo_model(img)

    MARGIN = 10
    countPerson = 0

    for (xmin, ymin, xmax, ymax, confidence, clas) in result.xyxy[0].tolist():
        if detectPose:
            roiPerson = img[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN]
        else:
            cv2.rectangle(img, (int(xmin) - MARGIN, int(ymin) - MARGIN), 
                          (int(xmax) + MARGIN, int(ymax) + MARGIN), (255, 0, 0), 2)

        countPerson += 1

    return countPerson, img

print("Carregando o modelo YOLOv5")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to('cpu') 

cap = cv2.VideoCapture(0)

while True:
    start = time.time()
    success, img = cap.read()

    if success:
        countPerson, frame = foundPeople(img, False)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    print("FPS: ", fps)
    cv2.imshow("Detecções", img)

    tecla = cv2.waitKey(1)
    if tecla == 27:
        break
cap.release()
cv2.destroyAllWindows()
