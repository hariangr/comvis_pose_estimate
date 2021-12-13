import cv2
from collections import deque

import mediapipe as mp
import time
import matplotlib.pyplot as plt


USE_CAMERA = False

# https://google.github.io/mediapipe/solutions/pose

xmin, xmax, ymin, ymax = 0, 100, -100, 100
# plt.axis = [xmin, xmax, ymin, ymax]

data_x = []  # Isinya frame number

hor_y = {
    i: [] for i in range(32 + 1)
}  # Isinya Map[Int -> Array[int]], dengan key adalah index dari landmark (0-32)
ver_y = {i: [] for i in range(32 + 1)}


def main():
    # region Source of Video
    if USE_CAMERA:
        camindex = 0
        cap = cv2.VideoCapture(camindex)
        if not cap.isOpened():
            raise Exception(f"Failed to open camera at index {camindex}")
    else:
        cap = cv2.VideoCapture("./sample/a.mp4")
    # endregion

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    i = 0

    while True:
        i += 1
        data_x.append(i)

        ok, frameRaw = cap.read()
        if not ok:
            raise Exception("Failed to read frame")

        frameCanvas = frameRaw.copy()

        frameRGB = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            mpDraw.draw_landmarks(
                frameCanvas, results.pose_landmarks, mpPose.POSE_CONNECTIONS
            )
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = frameCanvas.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)

                # if id == 0:
                hor_y[id].append(cy)
                ver_y[id].append(cx)

                cv2.circle(frameCanvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        cv2.imshow("Raw Frame", frameRaw)
        cv2.imshow("Image", frameCanvas)

        for id in hor_y.keys():
            plt.plot(
                deque(data_x, maxlen=30),
                deque(hor_y[id], maxlen=30),
                label=f"HOR{str(id)}",
            )
            plt.plot(
                deque(data_x, maxlen=30),
                deque(ver_y[id], maxlen=30),
                label=f"VER{str(id)}",
            )
        plt.pause(0.05)

        # plt.scatter

        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
