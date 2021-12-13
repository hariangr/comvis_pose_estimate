import cv2
from collections import deque

import mediapipe as mp
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


USE_CAMERA = False

# https://google.github.io/mediapipe/solutions/pose

hor_y = {
    i: deque([(0, 0)], maxlen=30) for i in range(32 + 1)
}  # Isinya Map[Int -> Array[Tupple(x_axis, y_axis)]], dengan key adalah index dari landmark (0-32), x_axis adalah nomer frame (horizontal axis), dan y_axis adalah valuenya
ver_y = {i: deque([(0, 0)], maxlen=30) for i in range(32 + 1)}


# def main():
#     # region Source of Video
#     if USE_CAMERA:
#         camindex = 0
#         cap = cv2.VideoCapture(camindex)
#         if not cap.isOpened():
#             raise Exception(f"Failed to open camera at index {camindex}")
#     else:
#         cap = cv2.VideoCapture("./sample/a.mp4")
#     # endregion

#     mpPose = mp.solutions.pose
#     pose = mpPose.Pose()
#     mpDraw = mp.solutions.drawing_utils

#     i = 0

#     while True:
#         i += 1
#         data_x.append(i)

#         ok, frameRaw = cap.read()
#         if not ok:
#             raise Exception("Failed to read frame")

#         frameCanvas = frameRaw.copy()

#         frameRGB = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2RGB)
#         results = pose.process(frameRGB)

#         if results.pose_landmarks:
#             mpDraw.draw_landmarks(
#                 frameCanvas, results.pose_landmarks, mpPose.POSE_CONNECTIONS
#             )
#             for id, lm in enumerate(results.pose_landmarks.landmark):
#                 h, w, c = frameCanvas.shape
#                 # print(id, lm)
#                 cx, cy = int(lm.x * w), int(lm.y * h)

#                 # if id == 0:
#                 hor_y[id].append(cy)
#                 ver_y[id].append(cx)

#                 cv2.circle(frameCanvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

#         cv2.imshow("Raw Frame", frameRaw)
#         cv2.imshow("Image", frameCanvas)

#         for id in hor_y.keys():
#             plt.plot(
#                 deque(data_x, maxlen=30),
#                 deque(hor_y[id], maxlen=30),
#                 label=f"HOR{str(id)}",
#             )
#             plt.plot(
#                 deque(data_x, maxlen=30),
#                 deque(ver_y[id], maxlen=30),
#                 label=f"VER{str(id)}",
#             )
#         plt.pause(0.05)

#         # plt.scatter

#         if cv2.waitKey(1) == ord("q"):
#             break


line_plots = {i: plt.plot(*zip(*hor_y[i]))[0] for i in range(32 + 1)}

cur_frame = 0



def animate(i):
    global cur_frame
    cur_frame += 1



    y = np.random.randn()
    hor_y[0].append((cur_frame, y))
    # ax.relim()
    # ax.autoscale_view()
    line_plots[0].set_data(*zip(*hor_y[0]))
    print(*zip(*hor_y[0]))


# (line,) = plt.plot(*zip(*data), c="black")

fig, ax = plt.subplots()
if __name__ == "__main__":
    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()
