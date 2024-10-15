import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

# cap = cv2.VideoCapture(0)

# with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,
#                     max_num_hands=6) as hands:
#     while cap.isOpened():
#         success, image = cap.read()
#         start = time.time()
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         image_height, image_width, _ = image.shape
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 try:
#                     cx, cy = mp_drawing._normalized_to_pixel_coordinates(
#                         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
#                         hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
#                         image_width, image_height)
#                     cv2.circle(image, (cx, cy), 4, [255, 0, 0], -1)
#                 except:
#                     cx=cy=0
#             end = time.time()
#             totalTime = end - start
#             fps = 1 / totalTime
#             cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
#                         (0, 255, 0), 2)
#             cv2.imshow('MediaPipe Hands', image)
#             if cv2.waitKey(5) & 0xFF == 27:
#                 break
def process_hands(hands, person_image, image, xmin, ymin, xmax, ymax):
    results = hands.process(person_image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image[ymin:ymax, xmin:xmax], hand_landmarks, mp_hands.HAND_CONNECTIONS)

    