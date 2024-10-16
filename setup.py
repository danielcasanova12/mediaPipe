import cv2
import mediapipe as mp
import numpy as np
import time

from facemesh import process_face_mesh
from handDetection import process_hands

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime = 0

def process_pose(pose, person_image, image):
    results = pose.process(person_image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def process_all(image_rgb, pose, face_mesh, hands, image):
    person_image = image_rgb

    # Processa face mesh
    process_face_mesh(face_mesh, person_image, image, 0, 0, image.shape[1], image.shape[0])
    # Processa pose
    process_pose(pose, person_image, image)
    # Processa mãos
    process_hands(hands, person_image, image, 0, 0, image.shape[1], image.shape[0])

with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4, min_tracking_confidence=0.4) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Não foi possível capturar a imagem.")
            break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        # Reduzir o tamanho da imagem para aliviar o processamento
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processar a pose, face mesh e mãos diretamente com MediaPipe
        process_all(image_rgb, pose, face_mesh, hands, image)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow("Detecção de Poses e MediaPipe", image)

        # Pressione 'ESC' para sair
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
