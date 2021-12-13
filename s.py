# Plotting stuff
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Computer Vision Stuff
import cv2
import mediapipe as mp

# Output to CSV stuff
import csv

# Label for each id
id_name = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


fig, (ax_ver, ax_hor) = plt.subplots(2)

# Rolling range, we don't want a really long plot
# Just show a small window of the latest n-th item
rolling_range = None

datas_hor = {id: deque([(0, 0)], maxlen=rolling_range) for id in range(32 + 1)}
datas_ver = {id: deque([(0, 0)], maxlen=rolling_range) for id in range(32 + 1)}
line_plots_hor = {
    id: ax_ver.plot(*zip(*datas_hor[id]), label=f"X {id}")[0] for id in range(32 + 1)
}
line_plots_ver = {
    id: ax_hor.plot(*zip(*datas_ver[id]), label=f"Y {id}")[0] for id in range(32 + 1)
}

# OpenCV stuff
cap = cv2.VideoCapture("./sample/Ellen_and_Brian.mp4")

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
    if cur_frame == 300:
        # Output the data to csv
        with open("Ellen_and_Brian.csv", "w", newline="") as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")

            # Write row header
            header = ["frame_no"]
            for i, col_name in id_name.items():
                header.append(f"x{col_name}")
                header.append(f"y{col_name}")
            writer.writerow(header)

            # Write value
            assert len(datas_hor) == len(datas_ver)
            for i in range(len(datas_hor[0])):
                row = [i] # First col is frame number
                for col_index, _ in id_name.items():
                    row.append(datas_hor[col_index][i][1])
                    row.append(datas_ver[col_index][i][1])
                writer.writerow(row)

        print("Finish 300 frame")
        return


if __name__ == "__main__":
    ani = animation.FuncAnimation(fig, animate, interval=1)
    plt.show()
