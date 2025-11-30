import cv2
import mediapipe as mp

# Configuración de MediaPipe (La "IA")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configurar la captura de video
# IMPORTANTE: Si tienes DroidCam, prueba con index 0, 1 o 2 hasta que salga imagen.
# 0 suele ser la webcam integrada, 1 suele ser DroidCam si tienes otra cámara conectada.
cap = cv2.VideoCapture(0) # <--- CAMBIA ESTE NÚMERO SI NO ABRE LA CÁMARA

print("Iniciando prueba de cámara... Presiona 'ESC' para salir.")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # Esto afina la detección en ojos y labios
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo obtener frame de la cámara. ¿Está conectada?")
            # Si usas DroidCam y esto falla, a veces ayuda reiniciar la app del cel
            break

        # Convertir a RGB (MediaPipe usa RGB, OpenCV usa BGR)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # --- AQUÍ OCURRE LA MAGIA ---
        results = face_mesh.process(image)

        # Volver a BGR para dibujar
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dibujamos la malla (tesselation) sobre tu cara
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                # Dibujamos los contornos (ojos, cara, boca)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

        # Voltear la imagen horizontalmente para que actúe como espejo (más natural)
        cv2.imshow('Prueba de Face Mesh - Presiona ESC para salir', cv2.flip(image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()