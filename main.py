import cv2
import mediapipe as mp
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor

from facemesh import process_face_mesh
from handDetection import process_hands

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

print("Carregando o modelo YOLOv5")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.to(device)
yolo_model.classes = [0] 
yolo_model.conf = 0.3

cap = cv2.VideoCapture(0)
pTime = 0

def process_pose(pose, person_image, image, xmin, ymin, xmax, ymax):
    results = pose.process(person_image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image[ymin:ymax, xmin:xmax], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

def process_with_threads(image_rgb, pose, face_mesh, hands, image, xmin, ymin, xmax, ymax):
    person_image = image_rgb[ymin:ymax, xmin:xmax]

    # Limitar o número de threads para evitar sobrecarga
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Processa face mesh em uma thread
        # executor.submit(process_face_mesh, face_mesh, person_image, image, xmin, ymin, xmax, ymax)
        # Processa pose em outra thread
        executor.submit(process_pose, pose, person_image, image, xmin, ymin, xmax, ymax)
        # Processa mãos em outra thread
        executor.submit(process_hands, hands, person_image, image, xmin, ymin, xmax, ymax)

with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=3, refine_landmarks=True, min_detection_confidence=0.4, min_tracking_confidence=0.4) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Não foi possível capturar a imagem.")
            break

        image = cv2.flip(image, 1)
        h, w, _ = image.shape

        # Reduzir o tamanho da imagem para aliviar o processamento
        image_small = cv2.resize(image, (640, 480))
        image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        results = yolo_model(image_rgb)

        # Ajustar o fator de escala para que as coordenadas sejam relativas ao tamanho original
        scale_x = w / 640
        scale_y = h / 480

        for detection in results.xyxy[0]:
            xmin, ymin, xmax, ymax, confidence, cls = detection

            # Reescalar as coordenadas para o tamanho original da imagem
            xmin, ymin, xmax, ymax = int(xmin * scale_x), int(ymin * scale_y), int(xmax * scale_x), int(ymax * scale_y)

            process_with_threads(image_rgb, pose, face_mesh, hands, image, xmin, ymin, xmax, ymax)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow("Detecção de Pessoas com YOLO e MediaPipe", image)

        # Pressione 'ESC' para sair
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
