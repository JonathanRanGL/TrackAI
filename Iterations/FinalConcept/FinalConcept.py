import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard
import time

# ==========================================
#   TRACKAI V14 - ANTI-LAG & THROTTLE
# ==========================================

# 1. OPTIMIZACIÓN DE INPUT
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

CAM_INDEX = 1

# --- COLORES DEL TEMA ---
COLOR_BG = (30, 30, 30)      
COLOR_ACCENT = (0, 255, 255) # Amarillo neón para visibilidad
COLOR_TEXT = (255, 255, 255) 

# 2. IA ULTRALIGERA
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def nothing(x): pass

# 3. CÁMARA: Solicitamos resolución baja a propósito para liberar USB
# Si DroidCam lo permite, esto forzará menos datos por el cable.
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- GUI ---
window_name = 'TrackAI V14 - Gaming Mode'
cv2.namedWindow(window_name)
cv2.resizeWindow(window_name, 500, 600)

# VALORES INICIALES (TU CONFIGURACIÓN)
cv2.createTrackbar('Ganancia X', window_name, 7, 200, nothing)
cv2.createTrackbar('Ganancia Y', window_name, 3, 100, nothing)
cv2.createTrackbar('Zona Muerta', window_name, 8, 50, nothing)
cv2.createTrackbar('Suavizado', window_name, 35, 100, nothing)
cv2.createTrackbar('Reactividad', window_name, 50, 200, nothing)

# Variables
paused = False
center_pitch, center_yaw = 0.0, 0.0
current_x, current_y = 0.0, 0.0 
gain_x, gain_y = 0.0, 0.0 

screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2
current_x, current_y = neutral_x, neutral_y

# VARIABLES PARA EL LIMITADOR DE MOUSE (THROTTLE)
last_mouse_time = 0
# Intervalo mínimo entre movimientos de mouse (en segundos)
# 0.016 = ~60 veces por segundo. 
# Si tienes lag, Python esperará este tiempo antes de mandar otra orden al mouse,
# dejando que la cámara procese video libremente.
MOUSE_INTERVAL = 0.010 

def draw_ui(img, fps):
    # Fondo simple
    cv2.rectangle(img, (0, 0), (640, 50), COLOR_BG, -1)
    
    # Estado y FPS
    status = "PAUSA" if paused else "ACTIVO"
    col = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(img, f"{status} | FPS: {int(fps)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    
    if not paused:
        cv2.putText(img, "[ESPACIO] Centrar", (350, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

print("--- MODO GAMING ACTIVADO ---")
print("Si los FPS caen, asegúrate de correr como ADMINISTRADOR.")

# Variables FPS
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # --- INPUT ---
    if keyboard.is_pressed('esc'): break
    if keyboard.is_pressed('p'):
        paused = not paused
        current_x, current_y = pydirectinput.position()
        time.sleep(0.3)

    # --- LÓGICA DE PAUSA ---
    if paused:
        final_img = cv2.flip(image, 1)
        draw_ui(final_img, 0)
        cv2.imshow(window_name, final_img)
        cv2.waitKey(1)
        continue

    # --- PROCESAMIENTO IA ---
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            for idx in [33, 263, 1, 61, 291, 199]:
                lm = face_landmarks.landmark[idx]
                x = lm.x * img_w
                y = lm.y * img_h
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles = cv2.RQDecomp3x3(rmat)[0]

            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # --- CONTROL ---
            if keyboard.is_pressed('space'):
                center_pitch = pitch
                center_yaw = yaw
                current_x, current_y = neutral_x, neutral_y

            # Parámetros
            gain_x = cv2.getTrackbarPos('Ganancia X', window_name) / 100.0
            gain_y = cv2.getTrackbarPos('Ganancia Y', window_name) / 100.0
            deadzone = cv2.getTrackbarPos('Zona Muerta', window_name) / 100.0
            
            min_alpha = cv2.getTrackbarPos('Suavizado', window_name) / 1000.0
            if min_alpha < 0.001: min_alpha = 0.001
            reactivity = cv2.getTrackbarPos('Reactividad', window_name) / 100.0

            # Deltas
            delta_yaw = yaw - center_yaw
            delta_pitch = pitch - center_pitch

            if abs(delta_yaw) < deadzone: delta_yaw = 0
            else: delta_yaw = delta_yaw - (deadzone if delta_yaw > 0 else -deadzone)

            if abs(delta_pitch) < deadzone: delta_pitch = 0
            else: delta_pitch = delta_pitch - (deadzone if delta_pitch > 0 else -deadzone)

            target_x = neutral_x + (delta_yaw * gain_x * 30) 
            target_y = neutral_y - (delta_pitch * gain_y * 30)

            # Filtro Dinámico
            dist = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            alpha = min(1.0, min_alpha + (dist * reactivity / 100.0))

            current_x = current_x + alpha * (target_x - current_x)
            current_y = current_y + alpha * (target_y - current_y)

            # --- THROTTLE DE MOUSE (LA SOLUCIÓN AL LAG) ---
            # Solo enviamos la orden al mouse si ha pasado suficiente tiempo.
            # Esto evita que saturemos la cola de eventos de Windows.
            current_time = time.time()
            if current_time - last_mouse_time > MOUSE_INTERVAL:
                pydirectinput.moveTo(int(current_x), int(current_y))
                last_mouse_time = current_time

    # FPS Calc
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Dibujar UI
    final_img = cv2.flip(image, 1)
    draw_ui(final_img, fps)
    
    cv2.imshow(window_name, final_img)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()