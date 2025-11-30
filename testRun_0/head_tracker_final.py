import cv2
import mediapipe as mp
import numpy as np
import pydirectinput
import keyboard  # Para detectar teclas globalmente
import time

# --- CONFIGURACIÓN DE SENSIBILIDAD (AJUSTA ESTO A TU GUSTO) ---
SENS_X = 15.0   # Qué tan rápido se mueve a los lados
SENS_Y = 10.0    # Qué tan rápido mira arriba/abajo (En ETS2 suele ser menos)
SMOOTHING = 0.5 # Entre 0.1 (muy suave/lento) y 0.9 (muy rápido/vibrante)

# --- CONFIGURACIÓN DE CÁMARA ---
CAM_INDEX = 1   # El índice que te funcionó

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(CAM_INDEX)

# Variables de estado
paused = False
center_y = 0
center_x = 0
prev_mouse_x, prev_mouse_y = 0, 0

# Obtener tamaño de pantalla para saber el centro
screen_w, screen_h = pydirectinput.size()
neutral_x, neutral_y = screen_w // 2, screen_h // 2

print("--- HEAD TRACKER INICIADO ---")
print(" [ESPACIO]  -> Recentrar vista (Mira al frente y presiona)")
print(" [P]        -> Pausar/Despausar el tracking")
print(" [ESC]      -> Salir")

pydirectinput.FAILSAFE = False # Evita que se cierre si tocas la esquina

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # 1. Gestión de Teclas
    if keyboard.is_pressed('esc'):
        print("Saliendo...")
        break
    
    if keyboard.is_pressed('p'):
        paused = not paused
        print(f"Pausa: {paused}")
        time.sleep(0.3) # Pequeño delay para no detectar doble pulsación

    # Si está pausado, solo mostramos la cámara pero no calculamos nada
    if paused:
        cv2.putText(image, "PAUSADO (Presiona 'P')", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Head Tracker', image)
        cv2.waitKey(1)
        continue

    # 2. Procesamiento de Imagen
    img_h, img_w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Usamos 3D para calcular rotación
            face_3d = []
            face_2d = []

            # Puntos clave: Nariz(1), Barbilla(199), Ojo I(33), Ojo D(263), Boca I(61), Boca D(291)
            for idx in [33, 263, 1, 61, 291, 199]:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Cámara Virtual para matemáticas
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Calcular rotación de la cabeza
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            # Guardamos el resultado en una variable genérica
            resultado = cv2.RQDecomp3x3(rmat)
            # Los ángulos siempre son el primer elemento (índice 0)
            angles = resultado[0]

            # Obtener ángulos actuales
            pitch = angles[0] * 360
            yaw = angles[1] * 360

            # 3. Lógica de Recentrado
            if keyboard.is_pressed('space'):
                center_x = yaw
                center_y = pitch
                print("¡Recentrado!")

            # 4. Calcular Delta (Diferencia entre donde miras y el centro)
            delta_x = (yaw - center_x) * SENS_X
            delta_y = (pitch - center_y) * SENS_Y

            # 5. Mover el Mouse (Con suavizado)
            # Invertimos X porque en tracking suele ser espejo
            target_x = int(neutral_x - delta_x) 
            target_y = int(neutral_y - delta_y) # Quita el menos si se invierte arriba/abajo

            # Interpolación simple para suavizar
            final_x = int(prev_mouse_x + (target_x - prev_mouse_x) * SMOOTHING)
            final_y = int(prev_mouse_y + (target_y - prev_mouse_y) * SMOOTHING)

            pydirectinput.moveTo(final_x, final_y)
            
            prev_mouse_x, prev_mouse_y = final_x, final_y

            # Visualización (Dibuja nariz para referencia)
            nose_pt = (int(face_landmarks.landmark[1].x * img_w), int(face_landmarks.landmark[1].y * img_h))
            cv2.circle(image, nose_pt, 5, (0, 255, 0), -1)

    cv2.imshow('Head Tracker', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()