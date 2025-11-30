import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard
import time

# --- AJUSTES DE SISTEMA ---
# Máxima prioridad al rendimiento
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

# Regresamos a la configuración que funcionaba bien (640x480 @ 60FPS)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- GUI DE ALTA PRECISIÓN ---
cv2.namedWindow('Head Tracker V12 - No Steps')
cv2.resizeWindow('Head Tracker V12 - No Steps', 500, 550)

# GANANCIAS (Slider 0-200)
# Valor sugerido: X=80, Y=50
cv2.createTrackbar('Ganancia X (x100)', 'Head Tracker V12 - No Steps', 80, 200, nothing)
cv2.createTrackbar('Ganancia Y (x100)', 'Head Tracker V12 - No Steps', 50, 100, nothing)

# ZONA MUERTA (Slider 0-50)
cv2.createTrackbar('Zona Muerta', 'Head Tracker V12 - No Steps', 5, 50, nothing)

# --- NUEVO CONTROL DE SUAVIZADO DINÁMICO ---
# Min Alpha: Suavizado base cuando estás casi quieto. 
# Súbelo si tiembla. bájalo si quieres más respuesta.
# Slider 10 = Alpha 0.010
cv2.createTrackbar('Suavizado Min (Estabilidad)', 'Head Tracker V12 - No Steps', 10, 100, nothing)

# Reactividad: Qué tan rápido suelta el filtro cuando te mueves rápido.
# Slider alto = Reacción instantánea. Slider bajo = Movimiento cinematográfico.
cv2.createTrackbar('Reactividad', 'Head Tracker V12 - No Steps', 50, 200, nothing)


# Variables globales
paused = False
center_pitch, center_yaw = 0.0, 0.0
current_x, current_y = 0.0, 0.0 
# Inicialización segura
gain_x, gain_y = 0.0, 0.0 

# Centro de pantalla
screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2
current_x, current_y = neutral_x, neutral_y

# FPS
prev_frame_time = 0
fps_display = 0

print("--- TRACKER V12: FLUJO DINÁMICO ---")
print(" [ESPACIO] -> Calibrar centro")
print(" [P]       -> Pausa")
print(" [ESC]     -> Salir")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Teclas
    if keyboard.is_pressed('esc'): break
    if keyboard.is_pressed('p'):
        paused = not paused
        curr_real_x, curr_real_y = pydirectinput.position()
        current_x, current_y = curr_real_x, curr_real_y
        time.sleep(0.3)

    if paused:
        cv2.putText(image, "PAUSA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Head Tracker V12 - No Steps', image)
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
            gain_x = cv2.getTrackbarPos('Ganancia X (x100)', 'Head Tracker V12 - No Steps') / 100.0
            gain_y = cv2.getTrackbarPos('Ganancia Y (x100)', 'Head Tracker V12 - No Steps') / 100.0
            deadzone = cv2.getTrackbarPos('Zona Muerta', 'Head Tracker V12 - No Steps') / 100.0
            
            # Parametros del Filtro Dinámico
            min_alpha = cv2.getTrackbarPos('Suavizado Min (Estabilidad)', 'Head Tracker V12 - No Steps') / 1000.0
            if min_alpha < 0.001: min_alpha = 0.001
            
            reactivity = cv2.getTrackbarPos('Reactividad', 'Head Tracker V12 - No Steps') / 100.0

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

            # 5. Objetivo (Target)
            # Uso el multiplicador base 30 que funcionó bien en V9/V10
            target_x = neutral_x + (delta_yaw * gain_x * 30) 
            target_y = neutral_y - (delta_pitch * gain_y * 30)

            # --- FILTRO DINÁMICO (ELIMINACIÓN DE SALTOS) ---
            # En lugar de detener el mouse si la distancia es pequeña (causa de saltos),
            # simplemente aplicamos un filtro MUY pesado (min_alpha).
            # Esto hace que el mouse se "deslice" infinitamente lento en lugar de saltar.
            
            dist = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # Fórmula de Alpha variable:
            # Si dist es 0 -> Alpha es min_alpha (Estable)
            # Si dist crece -> Alpha crece linealmente (Rápido)
            alpha = min(1.0, min_alpha + (dist * reactivity / 100.0))

            current_x = current_x + alpha * (target_x - current_x)
            current_y = current_y + alpha * (target_y - current_y)

            # Mover Mouse
            # Eliminé el "if dist > 0.3". Ahora movemos siempre.
            # El filtro se encarga de que no tiemble.
            pydirectinput.moveTo(int(current_x), int(current_y))

    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_display = 0.9 * fps_display + 0.1 * fps
    
    cv2.putText(image, f"FPS: {int(fps_display)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    cv2.imshow('Head Tracker V12 - No Steps', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()