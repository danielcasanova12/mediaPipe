#https://google.github.io/mediapipe/solutions/pose.html
#https://www.youtube.com/watch?v=brwgBf6VB0I
import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, upBody = False, smooth = True, enable_segmentation = False,
                       smooth_segmentation = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.enable_segmentation,
                                     self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        h, w, c = img.shape
        self.lmList = []
        x_max, y_max = 0,0
        x_min, y_min = 0,0
        offset = 10
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
               # print (id, lm)
               cx, cy = int(lm.x * w), int(lm.y * h)
               self.lmList.append([id, cx, cy])
               if id == 0:
                   x_max, y_max = cx, cy
                   x_min, y_min = cx, cy
               if cx > x_max:
                   x_max = cx
               elif cx < x_min:
                   x_min = cx
               if cy > y_max:
                   y_max = cy
               if cy < y_min:
                   y_min = cy
               if draw:
                  cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
            # if draw:
            #     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        x_min = x_min - offset if x_min - offset > 0 else 0
        y_min = y_min - offset if y_min - offset > 0 else 0
        x_max = x_max + offset if x_max + offset < w else w
        y_max = y_max + offset if y_max + offset < h else h

        return self.lmList, (x_min, y_min, x_max, y_max)

    def findAngle(self, img, p1, p2, p3, draw=True):
        _, x1, y1 = self.lmList[p1]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
        if angle < 0:
            angle += 360
        # print (angle)

        if draw:
            cv2.line(img, (x1, y1),(x2, y2),(255,255,255), 3)
            cv2.line(img, (x3, y3),(x2, y2),(255,255,255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
        return angle

    def distance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        return dist

    def getDistanceLandmarks(self, img, p1, p2, limiar=50, draw=True):
        _, x1, y1 = self.lmList[p1]
        _, x2, y2 = self.lmList[p2]

        dist = self.distance(x1,y1,x2,y2)

        if (y2 <= y1):
            newPoint = (x1,y2)
            original = (x1,y1)
        else:
            newPoint = (x2,y1)
            original = (x2,y2)

        dist2 = self.distance(original[0], original[1], newPoint[0], newPoint[1])

        if draw:
            cv2.line(img, (x1, y1),(x2, y2),(255,255,255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(dist)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))

            cv2.circle(img, newPoint, 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, newPoint, 15, (0, 255, 0), 2)
            cv2.line(img, original,newPoint,(255,255,0), 3)
            cv2.putText(img, str(int(dist2)), (newPoint[0]-50, newPoint[1]+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0))

            if dist2 > limiar:
                cv2.putText(img, "Correct your Posture!", (150,50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        return dist, dist2


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0

    detector = poseDetector()
    while True:
        success, img = cap.read()
        if success:
            img = detector.findPose(img)
            lmList = detector.findPosition(img)
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.imshow("image", img)
        if cv2.waitKey(1) == 27:
            exit()

if __name__ == "__main__":
    main()