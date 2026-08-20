import cv2
import mediapipe as mp
import numpy as np
import time
import pygame


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "models/face_landmarker.task"


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# EYE LANDMARKS
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks
MOUTH = [61, 291, 0, 17, 13, 14]


# ============================================================
# SETTINGS
# ============================================================

EAR_THRESHOLD = 0.22

MAR_THRESHOLD = 0.60

DROWSY_TIME = 2.0

YAWN_TIME = 1.0


# ============================================================
# EAR FUNCTION
# ============================================================

def calculate_ear(landmarks, eye_points):

    points = []

    for index in eye_points:

        landmark = landmarks[index]

        points.append(
            np.array([
                landmark.x,
                landmark.y
            ])
        )

    vertical_1 = np.linalg.norm(
        points[1] - points[5]
    )

    vertical_2 = np.linalg.norm(
        points[2] - points[4]
    )

    horizontal = np.linalg.norm(
        points[0] - points[3]
    )

    ear = (
        vertical_1 + vertical_2
    ) / (2 * horizontal)

    return ear

def calculate_mar(landmarks):

    # Mouth points
    left = np.array([
        landmarks[61].x,
        landmarks[61].y
    ])

    right = np.array([
        landmarks[291].x,
        landmarks[291].y
    ])

    top = np.array([
        landmarks[13].x,
        landmarks[13].y
    ])

    bottom = np.array([
        landmarks[14].x,
        landmarks[14].y
    ])

    # Mouth dimensions
    vertical = np.linalg.norm(
        top - bottom
    )

    horizontal = np.linalg.norm(
        left - right
    )

    mar = vertical / horizontal

    return mar
# ============================================================
# PYGAME ALARM
# ============================================================

pygame.mixer.init()

ALARM_FILE = "sounds/alarm.wav"

try:

    alarm_sound = pygame.mixer.Sound(ALARM_FILE)

except:

    alarm_sound = None

    print("WARNING: alarm.wav not found.")


alarm_playing = False


def start_alarm():

    global alarm_playing

    if alarm_sound is not None and not alarm_playing:

        alarm_sound.play(-1)

        alarm_playing = True


def stop_alarm():

    global alarm_playing

    if alarm_playing:

        pygame.mixer.stop()

        alarm_playing = False


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


# ============================================================
# VARIABLES
# ============================================================

closed_start_time = None

closed_time = 0

yawn_start_time = None
yawn_time = 0

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print("ERROR: Could not read camera.")

        break


    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe Image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    results = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # ========================================================
    # FACE DETECTED
    # ========================================================

    if results.face_landmarks:

        landmarks = results.face_landmarks[0]


        # ----------------------------------------------------
        # EAR
        # ----------------------------------------------------

        left_ear = calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        ear = (
            left_ear + right_ear
        ) / 2

        mar = calculate_mar(landmarks)

        if mar > MAR_THRESHOLD:

            if yawn_start_time is None:
                yawn_start_time = time.time()

            yawn_time = time.time() - yawn_start_time

        else:

            yawn_start_time = None
            yawn_time = 0


        if yawn_time >= YAWN_TIME:

            yawn_status = "YAWNING"

        else:

            yawn_status = "NORMAL"


        # ----------------------------------------------------
        # EYE STATUS
        # ----------------------------------------------------

        if ear < EAR_THRESHOLD:

            eye_status = "CLOSED"


            if closed_start_time is None:

                closed_start_time = time.time()


            closed_time = (
                time.time() -
                closed_start_time
            )


        else:

            eye_status = "OPEN"

            closed_start_time = None

            closed_time = 0


        # ----------------------------------------------------
        # DROWSINESS
        # ----------------------------------------------------

        if closed_time >= DROWSY_TIME:

            driver_status = "DROWSY"

            start_alarm()

        else:

            driver_status = "AWAKE"

            stop_alarm()


        # ====================================================
        # DRAW EYE LANDMARKS
        # ====================================================

        for index in LEFT_EYE + RIGHT_EYE:

            x = int(
                landmarks[index].x * width
            )

            y = int(
                landmarks[index].y * height
            )

            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )
        for index in MOUTH:
            x = int(landmarks[index].x * width)
            y = int(landmarks[index].y * height)
            cv2.circle(frame,(x, y),3,(255, 0, 0),-1)


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        cv2.putText(
            frame,
            f"Mouth: {yawn_status}",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Yawn Time: {yawn_time:.1f}s",
            (30, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Closed: {closed_time:.1f}s",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        if driver_status == "DROWSY":

            status_color = (0, 0, 255)

        else:

            status_color = (0, 255, 0)


        cv2.putText(
            frame,
            f"STATUS: {driver_status}",
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2
        )


        # ----------------------------------------------------
        # Big warning
        # ----------------------------------------------------

        if driver_status == "DROWSY":

            cv2.rectangle(
                frame,
                (0, 0),
                (width - 1, height - 1),
                (0, 0, 255),
                5
            )

            cv2.putText(
                frame,
                "!!! WAKE UP !!!",
                (width // 2 - 180, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )


    else:

        stop_alarm()

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "AI Driver Drowsiness Detection",
        frame
    )


    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

stop_alarm()

camera.release()

cv2.destroyAllWindows()

landmarker.close()

pygame.quit()