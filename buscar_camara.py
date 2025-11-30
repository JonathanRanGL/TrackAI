import cv2

print("🔎 Buscando cámaras (Índices 0 al 10)...")

for i in range(10): # Aumentamos el rango a 10
    # Probamos sin DSHOW primero, a veces DroidCam lo prefiere así
    cap = cv2.VideoCapture(i) 
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ ¡CÁMARA ENCONTRADA EN EL ÍNDICE {i}!")
            h, w, _ = frame.shape
            print(f"   Resolución: {w}x{h}")
        else:
            print(f"❌ Índice {i}: Detectado, pero pantalla negra (¿Está ocupada?).")
    cap.release()

print("🏁 Búsqueda terminada.")