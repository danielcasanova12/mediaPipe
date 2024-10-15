import cv2
import mediapipe as mp
import numpy as np
cap = cv2.VideoCapture(0)
mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_face_mesh(face_mesh, person_image, image, xmin, ymin, xmax, ymax):
    results = face_mesh.process(person_image)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image[ymin:ymax, xmin:xmax],
                landmark_list=landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
# with mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.4,
#                    min_tracking_confidence=0.5) as face_mesh:
#     while cap.isOpened():
#         status, image = cap.read()
#         if not status:
#             continue
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(image)
#         landmarks = results.multi_face_landmarks
#         h, w, _ = image.shape

#         black_image = np.zeros((h, w, 3), np.uint8)
#         if landmarks:
#             for landmarks in landmarks:
#                 drawing.draw_landmarks(image=black_image, landmark_list=landmarks,
#                                        connections=mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
#                                        connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0),
#                                        thickness = 1, circle_radius = 1))
#                 drawing.draw_landmarks(image=image, landmark_list=landmarks,
#                                        connections=mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
#                                        connection_drawing_spec=drawing.DrawingSpec(color=(0, 255, 0),
#                                                             thickness=1, circle_radius=1))
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         cv2.imshow('Output', black_image)
#         cv2.imshow('Entrada', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#     cap.release()
#     cv2.destroyAllWindows()
