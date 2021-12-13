from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import cv2
import mediapipe as mp

fig, (ax_ver, ax_hor) = plt.subplots(2)

datas_hor = {id: deque([(0, 0)], maxlen=30) for id in range(32 + 1)}
datas_ver = {id: deque([(0, 0)], maxlen=30) for id in range(32 + 1)}
line_plots_hor = {
    id: ax_ver.plot(*zip(*datas_hor[id]), label=f"X {id}")[0] for id in range(32 + 1)
}
line_plots_ver = {
    id: ax_hor.plot(*zip(*datas_ver[id]), label=f"Y {id}")[0] for id in range(32 + 1)
}

# OpenCV stuff
cap = cv2.VideoCapture("./sample/a.mp4")

# Pose Estimation
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def animate(cur_frame):
    # OpenCV
    ok, frameRaw = cap.read()
    if not ok:
        raise Exception("Failed to read frame")
    frameRGB = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2RGB)
    frameCanvas = frameRGB.copy()

    # Pose Estimating
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            frameCanvas, results.pose_landmarks, mpPose.POSE_CONNECTIONS
        )
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = frameCanvas.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Prepare data for plotting
            datas_hor[id].append((cur_frame, cx))
            datas_ver[id].append((cur_frame, cy))

            # Visualize estimated pose
            cv2.circle(frameCanvas, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Preview Frame
    cv2.imshow("RGB", frameRGB)
    cv2.imshow("Pose Estimated", frameCanvas)
    cv2.waitKey(1)

    # Plotting
    ax_ver.relim()
    ax_ver.autoscale_view()
    ax_hor.relim()
    ax_hor.autoscale_view()
    for id in range(32 + 1):
        line_plots_hor[id].set_data(*zip(*datas_hor[id]))
        line_plots_ver[id].set_data(*zip(*datas_ver[id]))

    if cur_frame == 1:
        # Remove the very first data point
        # We can't create empty line plot, so to get around this we create
        # a line plot with single placeholder item
        # But the placeholder item will skew the line plot toward the 0 value
        # causing the range to be 0-400ish, while the y is at around 300ish
        # so removing the placeholder will decrease the range of the plot
        # to be around 250ish to 350ish
        for i in range(32 + 1):
            datas_hor[i].popleft()
            datas_ver[i].popleft()


ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()
