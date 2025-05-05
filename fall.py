import cv2
from ultralytics import YOLO


def is_fall(kps):
  # Returns whether a fall was detected
  # Extract key point coordinates: hips and knees
    hip = kps[11][:2]
    knee = kps[13][:2]
  # Judgment condition 1
    condition1 = abs(hip[0] - knee[0]) > abs(hip[1] - knee[1])
  # Judgment condition 2
    shoulder = kps[5][:2]
    shoulder_r = kps[6][:2]
    condition2 = abs(shoulder[0] - shoulder_r[0]) < abs(shoulder[1] - shoulder_r[1])
    if condition1 or condition2:
        return True
    else:
        return False


class Fall:
    def __init__(self):
        self.model = YOLO('yolo11s-pose.pt')
        self.SKELETON = [# Define connections between key points in the human body
            (0, 1), (1, 3), (0, 2), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
#Performs pose estimation and fall detection on input frames
    def process(self, frame):
        img = frame.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model.predict(img_rgb)
        try:
            for result in results:
                # Extract key point coordinate data
                kps_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for person_kps in kps_data:
                    # Checking for falls
                    fall_state = is_fall(person_kps)
                    print(fall_state)
                    for x, y, c in person_kps:
                        if fall_state:
                            x = int(x)
                            y = int(y)
                            cv2.putText(img, "falldown", (x - 20, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break

                    # Drawing key points
                    for x, y, c in person_kps:
                        if c > 0.5:   # Confidence
                            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                    # Drawn from skeleton connections
                    for i, j in self.SKELETON:
                        xi, yi, ci = person_kps[i]
                        xj, yj, cj = person_kps[j]
                        if ci > 0.5 and cj > 0.5:
                            cv2.line(img, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2)
        except Exception:
            print("error")

        return img
