import cv2
import mediapipe as mp
 
# Inicjalizacja MediaPipe i kamery - GLOBALNIE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)
 
def get_fingers_status():
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        fingers_status = []

        # Kciuk - dodatkowo sprawdzimy orientację
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_x_diff = abs(thumb_tip.x - thumb_ip.x)
        thumb_y_diff = abs(thumb_tip.y - thumb_ip.y)
        thumb_horizontal = thumb_x_diff > thumb_y_diff * 1.5  # palec bardziej poziomo

        if thumb_tip.x < thumb_ip.x:
            thumb_extended = True
        else:
            thumb_extended = False
        fingers_status.append((thumb_extended, thumb_horizontal))

        # Pozostałe palce (index, middle, ring, pinky)
        for tip_id in [8, 12, 16, 20]:
            tip = hand_landmarks.landmark[tip_id]
            dip = hand_landmarks.landmark[tip_id - 2]
            fingers_status.append(tip.y < dip.y)

        return fingers_status  # [(thumb_extended, thumb_horizontal), index, middle, ring, pinky]

    return None

 
# Funkcja do zwolnienia kamery
def release_camera():
    cap.release()
    cv2.destroyAllWindows()
