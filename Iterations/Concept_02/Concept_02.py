import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard
import time

# --- AJUSTES DE SISTEMA ---
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

CAM_INDEX = 1

# --- IA ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def nothing(x): pass

# Configuración 640x480 a 60FPS (La más estable)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- GUI ---
cv2.namedWindow('Head Tracker V8 - Precision')
cv2.resizeWindow('Head Tracker V8 - Precision', 500, 500)

# ESCALAS CORREGIDAS (0 a 40)
# Ahora el valor del slider se divide entre 10.
# Ejemplo: Slider 20 = Ganancia 2.0 (Antes era como 20.0)
cv2.createTrackbar('Ganancia X', 'Head Tracker V8 - Precision', 15, 40, nothing)
cv2.createTrackbar('Ganancia Y', 'Head Tracker V8 - Precision', 10, 40, nothing)

# Zona Muerta (Slider 15 = 1.5 grados)
cv2.createTrackbar('Zona Muerta', 'Head Tracker V8 - Precision', 15, 50, nothing)

# Suavizado (Slider 20 = Alpha 0.2)
# Menor valor = Más suave/Lento. Mayor valor = Rápido/Tiembla.
cv2.createTrackbar('Suavizado', 'Head Tracker V8 - Precision', 10, 50, nothing)

# Variables
paused = False
center_pitch, center_yaw = 0.0, 0.0
current_x, current_y = 0.0, 0.0 

screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2
current_x, current_y = neutral_x, neutral_y

# FPS
prev_frame_time = 0
fps_display = 0

print("--- TRACKER V8: CALIBRACIÓN FINA ---")
print(" [ESPACIO] -> Calibrar")
print(" [P]       -> Pausa")
print(" [ESC]     -> Salir")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    if keyboard.is_pressed('esc'): break
    if keyboard.is_pressed('p'):
        paused = not paused
        time.sleep(0.3)

    if paused:
        cv2.putText(image, "PAUSA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Head Tracker V8 - Precision', image)
        cv2.waitKey(1)
        continue

    # Procesamiento
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            # Obtención de coordenadas sub-píxel
            for idx in [33, 263, 1, 61, 291, 199]:
                lm = face_landmarks.landmark[idx]
                x = lm.x * img_w
                y = lm.y * img_h
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # PnP
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles = cv2.RQDecomp3x3(rmat)[0]

            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # --- CONTROL DEL MOUSE ---

            # 1. Calibración
            if keyboard.is_pressed('space'):
                center_pitch = pitch
                center_yaw = yaw
                current_x, current_y = neutral_x, neutral_y

            # 2. Lectura GUI 
            # DIVISOR CORRECTO: Divido entre 10.0 para tener precisión decimal
            # Slider 20 -> 2.0 (Antes esto se multiplicaba por 10 extra, resultando en 20.0)
            gain_x = cv2.getTrackbarPos('Ganancia X', 'Head Tracker V8 - Precision') / 10.0
            gain_y = cv2.getTrackbarPos('Ganancia Y', 'Head Tracker V8 - Precision') / 10.0
            
            deadzone = cv2.getTrackbarPos('Zona Muerta', 'Head Tracker V8 - Precision') / 10.0
            
            # Suavizado: Slider 10 -> Alpha 0.1 (Muy suave)
            smooth_val = cv2.getTrackbarPos('Suavizado', 'Head Tracker V8 - Precision') / 100.0
            if smooth_val < 0.01: smooth_val = 0.01

            # 3. Deltas
            delta_yaw = yaw - center_yaw
            delta_pitch = pitch - center_pitch

            # 4. Soft Deadzone
            if abs(delta_yaw) < deadzone:
                delta_yaw = 0
            else:
                delta_yaw = delta_yaw - (deadzone if delta_yaw > 0 else -deadzone)

            if abs(delta_pitch) < deadzone:
                delta_pitch = 0
            else:
                delta_pitch = delta_pitch - (deadzone if delta_pitch > 0 else -deadzone)

            # 5. Objetivo (SIN MULTIPLICADORES EXTRAÑOS)
            # Solo Ganancia Pura.
            target_x = neutral_x + (delta_yaw * gain_x * 50) # *50 convierte grados a pixeles de forma razonable
            target_y = neutral_y - (delta_pitch * gain_y * 50)

            # 6. Filtro Exponencial Estándar
            current_x = current_x + smooth_val * (target_x - current_x)
            current_y = current_y + smooth_val * (target_y - current_y)

            # 7. ANTI-JITTER (Histéresis de Píxel)
            # Solo movemos el mouse si el cambio es mayor a 1 pixel.
            # Esto evita que el mouse vibre en el mismo lugar.
            mouse_x = int(current_x)
            mouse_y = int(current_y)
            
            # Obtenemos posición actual real del mouse para comparar
            curr_real_x, curr_real_y = pydirectinput.position()
            
            # Si la distancia al nuevo punto es menor a 2 pixeles, NO mover (estabilidad total)
            dist_to_move = np.sqrt((mouse_x - curr_real_x)**2 + (mouse_y - curr_real_y)**2)
            
            if dist_to_move > 1.5: 
                pydirectinput.moveTo(mouse_x, mouse_y)

    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_display = 0.9 * fps_display + 0.1 * fps
    
    cv2.putText(image, f"FPS: {int(fps_display)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow('Head Tracker V8 - Precision', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()