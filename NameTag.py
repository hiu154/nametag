import cv2
import mediapipe as mp
import yaml
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat


with open("settings.yml", "r") as f:
    config = yaml.safe_load(f)

name_tag = config["name_tag"]
font_scale = config["font_scale"]
font_thickness = config["font_thickness"]
box_offset_y = config["box_offset_y"]
box_padding = config["box_padding"]
font_color = tuple(config["font_color"])
box_color = tuple(config["box_color"])
box_alpha = config.get("box_alpha", 0.6)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30, fmt=PixelFormat.BGR) as cam:
    print(f"🎥 Virtual camera started: {cam.device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ids = [10, 151, 9]
                xs = [face_landmarks.landmark[i].x * w for i in ids]
                ys = [face_landmarks.landmark[i].y * h for i in ids]
                cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

                text_size = cv2.getTextSize(name_tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                box_width = text_size[0] + box_padding * 2
                box_height = text_size[1] + box_padding * 2
                box_x1 = cx - box_width // 2
                box_y1 = cy - box_offset_y - box_height - 10
                box_x2 = box_x1 + box_width
                box_y2 = box_y1 + box_height

                overlay = frame.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), box_color, -1)
                cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 1)

                text_x = box_x1 + (box_width - text_size[0]) // 2
                text_y = box_y1 + (box_height + text_size[1]) // 2 - 2
                cv2.putText(frame, name_tag, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

        cam.send(frame)
        cam.sleep_until_next_frame()

        
        cv2.imshow("MineCraft NameTag", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
