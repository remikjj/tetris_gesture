import cv2
import mediapipe as mp
import threading
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# Zmienne globalne
rotation_triggered = False
drop_triggered = False  # Nowa zmienna dla drop
last_hand_position = None
last_direction = None
gesture_state = "neutral"  # "neutral", "closed", "open", "thumb_bent"
gesture_cooldown = 0
COOLDOWN_TIME = 20  # ramki cooldownu po obrocie

# Nowe zmienne dla kontroli ruchu
last_direction_time = 0
direction_cooldown = 0
DIRECTION_COOLDOWN_TIME = 5  # ramki cooldownu dla kierunków
movement_triggered = False
last_significant_movement = None

def is_hand_closed(hand_landmarks):
    """Sprawdza czy dłoń jest zaciśnięta (pięść) na podstawie pozycji środkowego palca."""
    middle_finger_tip = hand_landmarks.landmark[12]
    middle_finger_mcp = hand_landmarks.landmark[9]
    
    # Dłoń jest zamknięta, jeśli czubek środkowego palca jest poniżej jego MCP
    return middle_finger_tip.y > middle_finger_mcp.y

def is_hand_open(hand_landmarks):
    """Sprawdza czy dłoń jest otwarta"""
    fingers_extended = []
    
    # Kciuk
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[3]
    fingers_extended.append(abs(thumb_tip.x - thumb_mcp.x) > 0.05)
    
    # Pozostałe palce
    for tip_id in [8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        fingers_extended.append(tip.y < pip.y - 0.02)
    
    # Dłoń jest otwarta jeśli większość palców jest wyprostowana
    extended_count = sum(fingers_extended)
    return extended_count >= 3

def is_thumb_bent_inward(hand_landmarks):
    """Sprawdza czy kciuk jest zgięty do środka dłoni przy wyprostowanych pozostałych palcach"""
    # Pozycje kciuka
    thumb_tip = hand_landmarks.landmark[4]      # czubek kciuka
    thumb_ip = hand_landmarks.landmark[3]       # staw międzypaliczkowy kciuka
    thumb_mcp = hand_landmarks.landmark[2]      # staw śródręczno-paliczkowy kciuka
    
    # Pozycje pozostałych palców
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    
    # Centrum dłoni
    palm_center = hand_landmarks.landmark[9]
    
    # Sprawdź czy pozostałe palce są wyprostowane
    fingers_extended = []
    for tip_id in [8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[tip_id - 2]
        fingers_extended.append(tip.y < pip.y - 0.02)
    
    # Przynajmniej 3 z 4 palców musi być wyprostowanych
    if sum(fingers_extended) < 3:
        return False
    
    # Sprawdź czy kciuk jest zgięty do środka
    # Kciuk jest zgięty jeśli czubek jest bliżej centrum dłoni niż podstawa
    thumb_to_palm_distance = ((thumb_tip.x - palm_center.x)**2 + (thumb_tip.y - palm_center.y)**2)**0.5
    thumb_base_to_palm_distance = ((thumb_mcp.x - palm_center.x)**2 + (thumb_mcp.y - palm_center.y)**2)**0.5
    
    # Dodatkowo sprawdź czy kciuk jest zgięty (czubek bliżej podstawy niż normalnie)
    thumb_bend_ratio = thumb_to_palm_distance / thumb_base_to_palm_distance
    
    return thumb_bend_ratio < 0.8  # Kciuk jest znacznie bliżej centrum niż podstawa

def camera_loop():
    global last_hand_position, last_direction, rotation_triggered, drop_triggered
    global gesture_state, gesture_cooldown
    global direction_cooldown, movement_triggered, last_significant_movement, last_direction_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Zmniejsz cooldowny
        if gesture_cooldown > 0:
            gesture_cooldown -= 1
        if direction_cooldown > 0:
            direction_cooldown -= 1

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_center = hand_landmarks.landmark[9]  # środek dłoni
            current_position = (hand_center.x, hand_center.y)

            # Sprawdź stan dłoni
            is_closed = is_hand_closed(hand_landmarks)
            is_open = is_hand_open(hand_landmarks)
            is_thumb_bent = is_thumb_bent_inward(hand_landmarks)
            
            # Logika przejść stanów
            if gesture_cooldown == 0:  # Tylko jeśli nie ma cooldownu
                if gesture_state == "neutral" and is_closed:
                    gesture_state = "closed"
                    print("Dłoń zaciśnięta")
                elif gesture_state == "closed" and is_open:
                    gesture_state = "open"
                    rotation_triggered = True
                    gesture_cooldown = COOLDOWN_TIME
                    print("Gest obrotu wykryty!")
                
                # DODAJ TEN FRAGMENT: Zezwól na drop z dowolnego stanu
                if is_thumb_bent:
                    gesture_state = "thumb_bent"
                    drop_triggered = True
                    gesture_cooldown = COOLDOWN_TIME
                    print("Gest drop wykryty! (kciuk zgięty)")
                
                # Reset do neutralnego stanu
                elif gesture_state in ["open", "thumb_bent"] and not is_open and not is_closed and not is_thumb_bent:
                    gesture_state = "neutral"

            # Wykrywanie ruchu dłoni z cooldownem (bez ruchu w dół dla drop)
            if last_hand_position and direction_cooldown == 0:
                dx = current_position[0] - last_hand_position[0]
                dy = current_position[1] - last_hand_position[1]
                threshold = 0.04  # Zwiększony próg dla bardziej wyraźnych gestów
                
                new_direction = None
                if abs(dx) > threshold or abs(dy) > threshold:
                    if abs(dx) > abs(dy):  # ruch poziomy dominuje
                        if dx > threshold:
                            new_direction = "right"
                        elif dx < -threshold:
                            new_direction = "left"
                    else:  # ruch pionowy dominuje
                        if dy < -threshold:  # tylko ruch w górę
                            new_direction = "up"
                        # Usunięto ruch w dół - teraz tylko kciuk służy do drop
                
                # Ustaw kierunek tylko jeśli się zmienił lub to pierwszy znaczący ruch
                if new_direction and new_direction != last_significant_movement:
                    last_direction = new_direction
                    last_significant_movement = new_direction
                    movement_triggered = True
                    direction_cooldown = DIRECTION_COOLDOWN_TIME
                    last_direction_time = time.time()
                    print(f"Nowy kierunek: {new_direction}")
                elif not new_direction:
                    # Reset gdy nie ma ruchu
                    if time.time() - last_direction_time > 0.5:  # Po 0.5s bez ruchu
                        last_direction = None
                        last_significant_movement = None
            
            # Zaktualizuj pozycję (zawsze)
            last_hand_position = current_position

            # Rysuj informacje na ekranie
            cv2.putText(frame, f"State: {gesture_state}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Direction: {last_direction}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Cooldown: {gesture_cooldown}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Dir Cooldown: {direction_cooldown}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Thumb bent: {is_thumb_bent}", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Rysuj punkty dłoni
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        else:
            # Reset gdy nie ma dłoni
            last_direction = None
            last_significant_movement = None
            if gesture_state != "neutral":
                gesture_state = "neutral"

        # Pokaż obraz
        cv2.imshow("Hand Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

def start_camera_thread():
    """Uruchom wątek kamery"""
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()
    print("Wątek kamery uruchomiony")

def get_latest_direction():
    """Pobierz ostatni wykryty kierunek ruchu"""
    return last_direction

def get_and_reset_rotation_triggered():
    """Sprawdź i zresetuj flagę obrotu"""
    global rotation_triggered
    if rotation_triggered:
        rotation_triggered = False
        return True
    return False

def get_and_reset_drop_triggered():
    """Sprawdź i zresetuj flagę drop"""
    global drop_triggered
    if drop_triggered:
        drop_triggered = False
        return True
    return False

def get_and_reset_movement_triggered():
    """Sprawdź i zresetuj flagę ruchu - użyj to gdy klocek wykonał ruch"""
    global movement_triggered
    if movement_triggered:
        movement_triggered = False
        return True
    return False

def reset_direction():
    """Resetuj kierunek - wywołaj gdy klocek spadnie lub zostanie umieszczony"""
    global last_direction, last_significant_movement, direction_cooldown
    last_direction = None
    last_significant_movement = None
    direction_cooldown = DIRECTION_COOLDOWN_TIME
    print("Kierunek zresetowany")

def release_camera():
    """Zwolnij zasoby kamery"""
    cap.release()
    cv2.destroyAllWindows()

# Test funkcji
if __name__ == "__main__":
    start_camera_thread()
    
    try:
        while True:
            # Sprawdź rotację
            if get_and_reset_rotation_triggered():
                print("🔄 OBRÓT WYKRYTY!")
            
            # Sprawdź drop
            if get_and_reset_drop_triggered():
                print("⬇️ DROP WYKRYTY! (kciuk zgięty)")
            
            # Sprawdź ruch
            if get_and_reset_movement_triggered():
                direction = get_latest_direction()
                if direction:
                    print(f"➡️ RUCH WYKRYTY: {direction}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Zakończenie...")
        release_camera()
