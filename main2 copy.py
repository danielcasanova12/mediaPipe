# import cv2
# import mediapipe as mp
# import time

# # Inicializa os módulos MediaPipe
# mp_holistic = mp.solutions.holistic
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # Configura a webcam
# cap = cv2.VideoCapture(0)

# # Variáveis para calcular o FPS
# prev_frame_time = 0
# new_frame_time = 0

# # Inicializa o Holistic
# with mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     enable_segmentation=True,
#     smooth_segmentation=True,
#     refine_face_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Falha ao capturar a imagem da webcam.")
#             break
        
#         # Converte a imagem de BGR para RGB para o MediaPipe
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Faz a detecção dos landmarks
#         results = holistic.process(image_rgb)
        
#         # Volta a imagem para BGR para exibir no OpenCV
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#         # Desenha os landmarks no frame
#         if results.face_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr, 
#                 results.face_landmarks, 
#                 mp_face_mesh.FACEMESH_TESSELATION)

#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr, 
#                 results.pose_landmarks, 
#                 mp_holistic.POSE_CONNECTIONS)

#         if results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr, 
#                 results.left_hand_landmarks, 
#                 mp_holistic.HAND_CONNECTIONS)

#         if results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image_bgr, 
#                 results.right_hand_landmarks, 
#                 mp_holistic.HAND_CONNECTIONS)

#         # Calcula o FPS
#         new_frame_time = time.time()
#         fps = 1 / (new_frame_time - prev_frame_time)
#         prev_frame_time = new_frame_time

#         # Exibe o FPS no frame
#         cv2.putText(image_bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         # Exibe a imagem processada
#         cv2.imshow('MediaPipe Holistic - FPS', image_bgr)

#         # Pressiona 'q' para sair
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# # Libera a webcam e fecha as janelas
# cap.release()
# cv2.destroyAllWindows()
