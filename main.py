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

from const import id_name

# OpenCV stuff
cap = cv2.VideoCapture("./sample/Ellen_and_Brian.mp4")
capB = cv2.VideoCapture("./sample/a.mp4")

fig, ((ax_ver, bx_ver, analyze_ver, overall_sim), (ax_hor, bx_hor, analyze_hor, _)) = plt.subplots(2, 4)

# Rolling range, we don't want a really long plot
# Just show a small window of the latest n-th item
rolling_range = 50

# For video A
datas_hor = {id: deque([(0, 0)] * rolling_range, maxlen=rolling_range) for id in range(32 + 1)}
datas_ver = {id: deque([(0, 0)] * rolling_range, maxlen=rolling_range) for id in range(32 + 1)}
ax_ver.title.set_text("Vid A Vertical Comp")
ax_hor.title.set_text("Vid A Horizontal Comp")
line_plots_hor = {
    id: ax_hor.plot(*zip(*datas_hor[id]), label=f"Vid A X {id}")[0]
    for id in range(32 + 1)
}
line_plots_ver = {
    id: ax_ver.plot(*zip(*datas_ver[id]), label=f"Vid A Y {id}")[0]
    for id in range(32 + 1)
}

# For video B
datas_hor_b = {id: deque([(0, 0)] * rolling_range, maxlen=rolling_range) for id in range(32 + 1)}
datas_ver_b = {id: deque([(0, 0)] * rolling_range, maxlen=rolling_range) for id in range(32 + 1)}
bx_ver.title.set_text("Vid B Vertical Comp")
bx_hor.title.set_text("Vid B Horizontal Comp")
line_plots_hor_b = {
    id: bx_hor.plot(*zip(*datas_hor_b[id]), label=f"Vid B X {id}")[0]
    for id in range(32 + 1)
}
line_plots_ver_b = {
    id: bx_ver.plot(*zip(*datas_ver_b[id]), label=f"Vid B Y {id}")[0]
    for id in range(32 + 1)
}

data_analyze_overall = []
data_analyze_ver = [0] * 50
data_analyze_hor = [0] * 50
analyze_ver.plot(data_analyze_ver)[0]
analyze_hor.plot(data_analyze_hor)[0]
analyze_ver.title.set_text("Vertical Similarity")
analyze_hor.title.set_text("Horizontal Similarity")
overall_sim.title.set_text("Overall Similarity")
analyze_ver.set_ylim([0, 100])
analyze_hor.set_ylim([0, 100])
overall_sim.set_ylim([0, 100])

# Pose Estimation
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def update_plot():
    # Plotting
    ax_ver.relim()
    ax_ver.autoscale_view()
    ax_hor.relim()
    ax_hor.autoscale_view()
    bx_ver.relim()
    bx_ver.autoscale_view()
    bx_hor.relim()
    bx_hor.autoscale_view()
    for id in range(32 + 1):
        line_plots_hor[id].set_data(*zip(*datas_hor[id]))
        line_plots_ver[id].set_data(*zip(*datas_ver[id]))
        line_plots_hor_b[id].set_data(*zip(*datas_hor_b[id]))
        line_plots_ver_b[id].set_data(*zip(*datas_ver_b[id]))


def analyze():
    # Analisa komponen horizontal
    total_hor = 0
    total_ver = 0
    for i, _ in enumerate(id_name):
        hor_vid_a = [data[1] for data in datas_hor[i]]
        hor_vid_b = [data[1] for data in datas_hor_b[i]]
        total_hor += np.corrcoef(hor_vid_a, hor_vid_b)

        ver_vid_a = [data[1] for data in datas_ver[i]]
        ver_vid_b = [data[1] for data in datas_ver_b[i]]
        total_ver += np.corrcoef(ver_vid_a, ver_vid_b)

    total_hor = total_hor[0][1] / len(id_name)
    total_hor = remap(total_hor, -1, 1, 0, 100)
    total_ver = total_ver[0][1] / len(id_name)
    total_ver = remap(total_ver, -1, 1, 0, 100)

    total_sim = (total_hor + total_ver) / 2
    
    data_analyze_overall.append(total_sim)
    overall_sim.plot(data_analyze_overall)

    data_analyze_hor.append(total_hor)
    data_analyze_ver.append(total_ver)

    analyze_ver.plot(data_analyze_ver)
    analyze_hor.plot(data_analyze_hor)


def animate(cur_frame):
    # OpenCV
    ok, frameRaw = cap.read()
    okB, frameRaw_b = capB.read()

    if not ok or not okB:
        print("Failed to read frame")
        return
    frameRGB = cv2.cvtColor(frameRaw, cv2.COLOR_BGR2RGB)
    frameCanvas = frameRGB.copy()

    frameRGB_b = cv2.cvtColor(frameRaw_b, cv2.COLOR_BGR2RGB)
    frameCanvas_b = frameRGB_b.copy()

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

    results_b = pose.process(frameRGB_b)
    if results_b.pose_landmarks:
        mpDraw.draw_landmarks(
            frameCanvas_b, results_b.pose_landmarks, mpPose.POSE_CONNECTIONS
        )
        for id, lm in enumerate(results_b.pose_landmarks.landmark):
            h, w, c = frameCanvas_b.shape
            cx, cy = int(lm.x * w), int(lm.y * h)

            # Prepare data for plotting
            datas_hor_b[id].append((cur_frame, cx))
            datas_ver_b[id].append((cur_frame, cy))

            # Visualize estimated pose
            cv2.circle(frameCanvas_b, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Preview Frame
    cv2.imshow("Pose Estimated Vid A", frameCanvas)
    cv2.imshow("Pose Estimated Vid B", frameCanvas_b)
    cv2.waitKey(1)

    update_plot()
    if cur_frame > rolling_range:
        analyze()

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
    # if cur_frame == 280:
    #     # Output the data to csv
    #     with open("a.csv", "w", newline="") as file:
    #         writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")

    #         # Write row header
    #         header = ["frame_no"]
    #         for i, col_name in id_name.items():
    #             header.append(f"x{col_name}")
    #             header.append(f"y{col_name}")
    #         writer.writerow(header)

    #         # Write value
    #         assert len(datas_hor) == len(datas_ver)
    #         for i in range(len(datas_hor[0])):
    #             row = [i]  # First col is frame number
    #             for col_index, _ in id_name.items():
    #                 row.append(datas_hor[col_index][i][1])
    #                 row.append(datas_ver[col_index][i][1])
    #             writer.writerow(row)

    #     print("Finish 300 frame")
    #     return

def remap(OldValue, OldMin, OldMax, NewMin, NewMax):
    return (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin



if __name__ == "__main__":
    ani = animation.FuncAnimation(fig, animate, interval=5)
    plt.show()


