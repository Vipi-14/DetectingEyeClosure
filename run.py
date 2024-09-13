import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO("best.pt")
names = model.names

cap = cv2.VideoCapture("WhatsApp Video 2024-09-13 at 10.12.46 AM.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize variables for eye closure detection
eye_closed_frames = 0
eye_closed_threshold_seconds = 1  # Threshold in seconds
eye_closed_threshold_frames = eye_closed_threshold_seconds * fps  # Convert seconds to frames

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, show=False)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    annotator = Annotator(im0, line_width=2, example=names)

    eye_closed = False  # Flag to check if the eye is closed in the current frame

    if boxes is not None:
        for box, cls in zip(boxes, clss):
            clsName = names[int(cls)]
            xmax = int(box[0])
            ymin = int(box[1])
            xmin = int(box[2])
            ymax = int(box[3])

            # Set color based on the class name
            if clsName == 'closed':
                clr = (0, 0, 255)
                eye_closed = True  # Mark eye as closed
            elif clsName == 'opened':
                clr = (0, 255, 0)

            # Draw the bounding box and label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            tw, th = cv2.getTextSize(clsName, font, font_scale, font_thickness)[0]

            cv2.rectangle(im0, (xmin, ymin), (xmax, ymax), color=clr, thickness=2)
            cv2.putText(im0, clsName, (xmax, ymin - 5), font, font_scale, color=clr, thickness=font_thickness)

    # Check for eye closure duration
    if eye_closed:
        eye_closed_frames += 1
    else:
        eye_closed_frames = 0  # Reset counter if the eye is not closed

    # Display warning if eye has been closed for more than the threshold
    if eye_closed_frames > eye_closed_threshold_frames:
        print("Warning: Eye has been closed for more than 2 seconds!")
        cv2.putText(im0, "WARNING: Eye closed for more than 2 seconds!", (10, 10), font, 0.8 , (0, 0, 255), font_thickness)
    else:
        # Clear the warning when the eye is opened
        warning_triggered = False

    video_writer.write(im0)

cap.release()
video_writer.release()
