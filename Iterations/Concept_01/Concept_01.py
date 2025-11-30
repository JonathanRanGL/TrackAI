import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard
import time

# --- CONFIGURACIÓN DE RENDIMIENTO ---
# Desactivo el freno de seguridad de PyDirectInput para eliminar el stuttering
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

# Índice de mi cámara
CAM_INDEX = 1

# --- INICIALIZACIÓN IA ---
mp_face_mesh = mp.solutions.face_mesh
# Configuro parámetros bajos de confianza para priorizar velocidad (FPS) sobre precisión extrema
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def nothing(x): pass

# Configuración de cámara para intentar forzar 60FPS y baja resolución
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- GUI DE AJUSTES ---
cv2.namedWindow('Head Tracker V5 - Final')
cv2.resizeWindow('Head Tracker V5 - Final', 500, 450)

# Rangos ajustados de 0 a 40 para tener control fino (Valor real = Trackbar / 10)
# Inicializo X en 25 (2.5) y Y en 20 (2.0) como definí en mis pruebas
cv2.createTrackbar('Ganancia X', 'Head Tracker V5 - Final', 250, 400, nothing)
cv2.createTrackbar('Ganancia Y', 'Head Tracker V5 - Final', 200, 400, nothing)

# Deadzone para ignorar mi respiración o pulso (Valor / 100)
cv2.createTrackbar('Zona Muerta', 'Head Tracker V5 - Final', 5, 50, nothing)

# "Agresividad" del filtro dinámico. 
# Valor alto = Reacciona más rápido pero puede temblar un poco.
# Valor bajo = Más suave pero con más "peso".
cv2.createTrackbar('Reaccion Dinamica', 'Head Tracker V5 - Final', 40, 100, nothing)

# Variables globales de estado
paused = False
center_pitch, center_yaw = 0, 0
current_x, current_y = 0.0, 0.0 # Uso float para precisión sub-pixel en el filtro

# Centro de mi monitor
screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2
current_x, current_y = neutral_x, neutral_y

# Variables para suavizado de FPS
prev_frame_time = 0
fps_display = 0

print("--- SISTEMA LISTO ---")
print(" [ESPACIO] -> Calibrar centro")
print(" [P]       -> Pausa")
print(" [ESC]     -> Salir")

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # --- LÓGICA DE TECLADO ---
    if keyboard.is_pressed('esc'): break
    if keyboard.is_pressed('p'):
        paused = not paused
        time.sleep(0.3)

    if paused:
        cv2.putText(image, "PAUSA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Head Tracker V5 - Final', image)
        cv2.waitKey(1)
        continue

    # --- VISIÓN POR COMPUTADORA ---
    # Convertir color y procesar malla
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            img_h, img_w, _ = image.shape
            face_3d = []
            face_2d = []

            # Extraigo solo los puntos clave para mantener el rendimiento alto
            # Nariz(1), Barbilla(199), Ojos(33, 263), Boca(61, 291)
            for idx in [33, 263, 1, 61, 291, 199]:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Matemáticas de proyección (PnP)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2], [0, focal_length, img_w/2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            # Obtengo ángulos de Euler
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles = cv2.RQDecomp3x3(rmat)[0]

            # Multiplico por 360 para tener grados aproximados
            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # --- LÓGICA DE CONTROL DEL MOUSE ---

            # 1. Calibración
            if keyboard.is_pressed('space'):
                center_pitch = pitch
                center_yaw = yaw
                current_x, current_y = neutral_x, neutral_y

            # 2. Lectura de parámetros GUI
            gain_x = cv2.getTrackbarPos('Ganancia X', 'Head Tracker V5 - Final') / 10.0
            gain_y = cv2.getTrackbarPos('Ganancia Y', 'Head Tracker V5 - Final') / 10.0
            deadzone = cv2.getTrackbarPos('Zona Muerta', 'Head Tracker V5 - Final') / 100.0
            dynamic_factor = cv2.getTrackbarPos('Reaccion Dinamica', 'Head Tracker V5 - Final') / 100.0
            if dynamic_factor < 0.01: dynamic_factor = 0.01

            # 3. Deltas
            delta_yaw = yaw - center_yaw
            delta_pitch = pitch - center_pitch

            # 4. Zona Muerta (Deadzone)
            if abs(delta_yaw) < deadzone: delta_yaw = 0
            if abs(delta_pitch) < deadzone: delta_pitch = 0

            # 5. Cálculo del Objetivo (Target)
            # CORREGIDO: Cambié el signo de resta a suma en X para invertir el eje.
            # Ahora: Cabeza Izq -> Mouse Izq.
            target_x = neutral_x + (delta_yaw * gain_x) 
            target_y = neutral_y - (delta_pitch * gain_y)

            # --- FILTRO DINÁMICO NO LINEAL (LA SOLUCIÓN AL LAG) ---
            # Calculo la distancia entre donde está el mouse y donde debería estar
            dist = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            # MAGIA MATEMÁTICA:
            # Si la distancia es grande (movimiento rápido), 'alpha' tiende a 1.0 (instantáneo).
            # Si la distancia es pequeña (temblor), 'alpha' tiende a 0.0x (muy suave).
            # El divisor 50.0 ajusta la curva de respuesta.
            alpha = min(1.0, (dist / 50.0) * dynamic_factor + 0.05)

            # Aplicar suavizado con el alpha variable
            current_x = current_x + alpha * (target_x - current_x)
            current_y = current_y + alpha * (target_y - current_y)

            # Mover mouse (solo redondeo al final)
            pydirectinput.moveTo(int(current_x), int(current_y))

    # --- CÁLCULO DE FPS SUAVIZADO ---
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    # Promedio simple: 90% el valor anterior, 10% el nuevo (evita parpadeo)
    fps_display = 0.9 * fps_display + 0.1 * fps
    
    # Mostrar FPS
    color = (0, 255, 0) if fps_display > 25 else (0, 0, 255)
    cv2.putText(image, f"FPS: {int(fps_display)}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    cv2.imshow('Head Tracker V5 - Final', image)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()