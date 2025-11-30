import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard
import time

# --- AJUSTES DE SISTEMA ---
# Desactivo la pausa de seguridad para máxima fluidez
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

CAM_INDEX = 1

# --- IA ---
mp_face_mesh = mp.solutions.face_mesh
# Configuración optimizada para velocidad sobre precisión extrema
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def nothing(x): pass

# Configuración 640x480 @ 60FPS (El punto óptimo rendimiento/latencia)
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- GUI DE ALTA PRECISIÓN ---
cv2.namedWindow('Head Tracker V10 - Final')
cv2.resizeWindow('Head Tracker V10 - Final', 500, 500)

# GANANCIAS EN CENTÉSIMAS (FLOAT)
# X: Rango 0.00 a 2.00. Valor sugerido: 80 (0.80)
cv2.createTrackbar('Ganancia X (x100)', 'Head Tracker V10 - Final', 80, 200, nothing)

# Y: Rango 0.00 a 1.00. Valor sugerido: 50 (0.50)
cv2.createTrackbar('Ganancia Y (x100)', 'Head Tracker V10 - Final', 50, 100, nothing)

# ZONA MUERTA (x100) - Slider 5 = 0.05 grados.
cv2.createTrackbar('Zona Muerta', 'Head Tracker V10 - Final', 5, 50, nothing)

# SUAVIZADO (Filtro Alpha x1000) - Valor sugerido: 30
# Menor valor = Más pesado/suave. Mayor valor = Más reactivo.
cv2.createTrackbar('Suavizado', 'Head Tracker V10 - Final', 30, 200, nothing)

# Variables globales
paused = False
center_pitch, center_yaw = 0.0, 0.0
current_x, current_y = 0.0, 0.0 
# Inicializo estas variables para evitar el NameError si no detecta cara al inicio
gain_x, gain_y = 0.0, 0.0 

# Centro de pantalla
screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2
current_x, current_y = neutral_x, neutral_y

# FPS
prev_frame_time = 0
fps_display = 0

print("--- TRACKER V10: VERSIÓN DEFINITIVA ---")
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
        # Sincronizo mi posición virtual con la real al pausar/despausar
        curr_real_x, curr_real_y = pydirectinput.position()
        current_x, current_y = curr_real_x, curr_real_y
        time.sleep(0.3)

    if paused:
        cv2.putText(image, "PAUSA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Head Tracker V10 - Final', image)
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

            # Coordenadas Sub-Píxel (Float) para máxima suavidad
            for idx in [33, 263, 1, 61, 291, 199]:
                lm = face_landmarks.landmark[idx]
                x = lm.x * img_w
                y = lm.y * img_h
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Matemáticas PnP
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

            # 2. Lectura GUI (Precisión Centésimas)
            gain_x = cv2.getTrackbarPos('Ganancia X (x100)', 'Head Tracker V10 - Final') / 100.0
            gain_y = cv2.getTrackbarPos('Ganancia Y (x100)', 'Head Tracker V10 - Final') / 100.0
            
            # Zona muerta ultra fina
            deadzone = cv2.getTrackbarPos('Zona Muerta', 'Head Tracker V10 - Final') / 100.0
            
            # Suavizado con alta granularidad (divisor 1000)
            smooth_val = cv2.getTrackbarPos('Suavizado', 'Head Tracker V10 - Final') / 1000.0
            if smooth_val < 0.005: smooth_val = 0.005

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

            # 5. Objetivo (Target) - Multiplicador base 30
            target_x = neutral_x + (delta_yaw * gain_x * 30) 
            target_y = neutral_y - (delta_pitch * gain_y * 30)

            # 6. Filtro Exponencial (Low Pass)
            current_x = current_x + smooth_val * (target_x - current_x)
            current_y = current_y + smooth_val * (target_y - current_y)

            # 7. ANTI-JITTER REFINADO (Micro-movimientos)
            # He bajado el umbral a 0.3 píxeles.
            # Solo filtra el ruido eléctrico puro. Los movimientos lentos y finos
            # ahora son manejados por el filtro de suavizado, no por un stop brusco.
            mouse_x = int(current_x)
            mouse_y = int(current_y)
            
            curr_real_x, curr_real_y = pydirectinput.position()
            dist_to_move = np.sqrt((mouse_x - curr_real_x)**2 + (mouse_y - curr_real_y)**2)
            
            # Umbral reducido para mayor finura
            if dist_to_move > 0.3: 
                pydirectinput.moveTo(mouse_x, mouse_y)
            else:
                # Si no movemos, mantenemos la posición virtual sincronizada con la real
                # para evitar "acumulación" de error.
                current_x, current_y = curr_real_x, curr_real_y

    # FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps_display = 0.9 * fps_display + 0.1 * fps
    
    cv2.putText(image, f"FPS: {int(fps_display)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # Ahora sí puedo mostrar esto sin error porque las variables están inicializadas
    cv2.putText(image, f"GX: {gain_x:.2f} GY: {gain_y:.2f}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
    
    cv2.imshow('Head Tracker V10 - Final', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()