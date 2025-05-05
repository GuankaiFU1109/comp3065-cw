import streamlit as st
import cv2
import numpy as np
import tempfile

import util
from SORT import Sort
from detect import Detector
from fall import Fall

# load coco.class for YOLO model object recognition
with open('coco.class', 'r') as f:
    classes = f.read().strip().split('\n')

# Initialize the detector and fall recognition module
detector = Detector()
fall = Fall()

# Initialize SORT tracker
sort_max_age = 5
sort_min_hits = 2
sort_iou_thresh = 0.2
sort_tracker = Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)

# frame handling functions
def processing_frame(frame,fps):
    # Object detection
    img = frame.copy()
    outputs = detector.detect(img)
    # Fall detection
    img = fall.process(img)
    dets_to_sort = np.empty((0, 6))
    y_all = 0.000017
    y_all1 = 0.000017
    account = 0.000001
    car_count = 0.000001
    pedestrian_count = 0
    car_count = 0
    for box in outputs:
        box = box.cpu().numpy()
        x1, y1, x2, y2, score, index = box
        class_name = classes[int(index)]
        if index == 0:# Pedestrian (person)
            pedestrian_count += 1# Persons counted
            if x2 - x1 > y2 - y1:
                label = f"{class_name} {score:.2f} falldown"
                util.draw_red_box(img, box, label)
                continue
            else:
                label = f"{class_name} {score:.2f}"
                y_all += (y2-y1)
                account += 1
            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, score, index])))
            cv2.putText(img, "PEDESTRIAN DETECTED", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

        else :# Car
            if index == 2:
                y_all1 += (y2 - y1)
                car_count += 1
                label = f"{class_name} {score:.2f}"
                util.draw_red_box(img, box, label)
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, score, index])))
                cv2.putText(img, "CAR DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                continue
            else:
                continue
        util.draw_box(img, box, label)

    # Display of person and car counts
    cv2.putText(img, f"Total Pedestrians: {pedestrian_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Total Cars: {car_count}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    y_avg = y_all / account
    y_avg1 = y_all1 / (car_count if car_count > 0 else 1)

    # SORT tracker update, return tracker box with ID
    tracked_dets = sort_tracker.update(dets_to_sort)
    tracks = sort_tracker.getTrackers()
    for track in tracks:
        detclass = track.detclass
        for i, _ in enumerate(track.centroidarr):
            if i < len(track.centroidarr) - 1:
                point1 = (int(track.centroidarr[i][0]), int(track.centroidarr[i][1]))
                point2 = (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1]))

                # Draw the track line
                color = (255, 0, 0) if detclass == 0 else (0, 0, 255)
                cv2.line(img, point1, point2, color, thickness=2)

                if i == len(track.centroidarr) - 2:
                    p_distance = util.calculate_distance_numpy(point1, point2)

                    if detclass == 0:  # person
                        distance = (p_distance / y_avg) * 1.7
                        speed = distance * fps
                        cv2.putText(img, f"Person Speed: {speed:.2f} m/s", point2,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

                    elif detclass == 2:  # car
                        distance = (p_distance / y_avg1) * 1.5
                        speed = distance * fps
                        cv2.putText(img, f"Car Speed: {speed:.2f} m/s", point2,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    return img


def clear_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
# Streamlit main function: building a front-end interaction platform
def main():
    st.title("Video Processing Platform")

    if "has_rerun" not in st.session_state:
        st.session_state.has_rerun = False

    # Clear button
    if st.button("Clear all output"):
        clear_all()

    # Upload a video
    st.subheader("1. Upload a video")
    uploaded_file = st.file_uploader("Select a video file", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Create a temporary file to store uploaded videos
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_video_path = tfile.name

        # Show original video
        st.subheader("2.Show original video")
        video_file = open(input_video_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

        # Video Processing button
        st.subheader("3. Video Processing")

        # process button
        if st.button("Click to process video"):
            # Open the video file
            cap = cv2.VideoCapture(input_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Create temporary files to store processed videos
            output_video_path = 'p.mp4'

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Frame-by-frame processing of video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = processing_frame(frame, fps)
                out.write(processed_frame)

            # Release resources
            cap.release()
            out.release()

            # Display processed video
            st.subheader("4.Display processed video")
            with open(output_video_path, "rb") as f:
                video_bytes_processed = f.read()
                st.video(video_bytes_processed)

            # Provide downloads of processed videos
            with open(output_video_path, "rb") as f:
                st.download_button("Download processed video", f, file_name="processed_video.mp4")


if __name__ == "__main__":
    main()