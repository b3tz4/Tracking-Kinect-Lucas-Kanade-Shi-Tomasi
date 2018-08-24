from cv2 import*
import numpy as np
import freenect
import time

#Tomamos el tiempo Inicial
tiempoInicial = int(round(time.time(), 10))

#Inicializamos la Camara RGB de la Kinect
def get_video():
	array,_ = freenect.sync_get_video()
	array = cvtColor(array, COLOR_RGB2BGR)
	return array
#Inicializamos la Camara de Profundidad
def get_depth():
	array, _ = freenect.sync_get_depth()
	array = array.astype(np.uint8)
	return array

# Parametros para la funcion de Lucas Kanade
lk_params = dict( winSize  = (15,15),
    maxLevel = 2,
    criteria = (TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT, 10, 0.03))

#Capturamos una imagen y la convertimos de RGB a HSV
imagen_anterior = get_video()
hsv = cvtColor(imagen_anterior, COLOR_BGR2HSV)

#Convertimos la imagen a gris para poder introducirla en el bucle principal
frame_anterior = cvtColor(imagen_anterior, COLOR_BGR2GRAY)

#Establecemos el rango de colores que vamos a detectar
blancos_bajos = np.array([64, 49, 100], dtype=np.uint8)
blancos_altos = np.array([205, 144, 158], dtype=np.uint8)

#Crear una mascara con solo los pixeles dentro del rango de Blanco
mask = inRange(hsv, blancos_bajos, blancos_altos)

#Eliminamos ruido
kernel = np.ones((9,9),np.uint8)
mask = morphologyEx(mask, MORPH_OPEN,kernel)
mask = morphologyEx(mask, MORPH_CLOSE,kernel)
#mask = erode (mask, kernel, iterations = 1)
#mask = dilate (mask, kernel,iterations = 1)


#Detectamos contornos, nos quedamos con el mayor y calculamos su centro
_,contours, hierarchy = findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE)
mayor_contorno = max(contours, key = contourArea)
momentos = moments(mayor_contorno)
cx = float(momentos['m10']/momentos['m00'])
cy = float(momentos['m01']/momentos['m00'])
p0 = np.array([[[cx,cy]]],np.float32)
"""
#Cuando exista un Error en la Deteccion de Contornos Entonces Activar Shi Tomasi
feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
p0 = goodFeaturesToTrack(frame_anterior, mask = None, **feature_params)
"""

#Convertimos el punto elegido a un array de numpy que se pueda pasar como parametro
#a la funcion calcOpticalFlowPyrLK()


#Crea Otra Mascara para trazar la Linea de Tracking
mask_anterior = np.zeros_like(imagen_anterior)

while(1):
	
	#Tomamos el Tiempo Real
	tiempoActual = int(round(time.time(), 10)) - tiempoInicial
	
	#Difinimos cada cuanto segundos va a tomar la Foto
	tiempo_mod= int(tiempoActual % 10)
	tiempo_str= str(tiempoActual)
	
	#Capturamos una imagen y la convertimos de RGB a GRIS
	imagen = get_video()
	frame_gray = cvtColor(imagen, COLOR_BGR2GRAY)
	
	#Se aplica el metodo de Lucas Kanade
	p1, st, err = calcOpticalFlowPyrLK(frame_anterior, frame_gray, p0, None, **lk_params)

	# Elegir los puntos adecuados
	good_old  = p0[st==1]
	good_new  = p1[st==1]

	#Pintamos el centro con Rojo y la Liniea azul dibuja su Trayectoria
	for i,(new,old) in enumerate(zip(good_new, good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask_anterior = line(mask_anterior, (a,b),(c,d), (255, 0, 0), 1)
		imagen = circle(imagen,(a,b),2, (0,0,255),-1)
	
	#incluimos los mismos Parametros de Dibujo
	img = add(imagen,mask_anterior)
	
	#Actualiza el frame_anterior y el puntito del Centro en la Figura
	frame_anterior = frame_gray.copy()
	p0 = good_new.reshape(-1,1,2)
	
	#Mostramos la imagen original con la marca del centro
	imshow('Camara_Tiempo_Real', img)
	imshow('Mascara_Tiempo_Real', mask_anterior)
	#imshow('Frame', imagen)	
	imshow('Mascara_Inicial', mask)
	
	#Guarda la Imagen en la Direccion de Memoria Indicada
	if tiempo_mod == 0:
		print "Guardando Imagen" + tiempo_str + ".png"
		imwrite("/home/betza/UASLP/Imagen"+ tiempo_str +".png", img)
	
	#Finalizar Proceso con la Tecla ESC
	tecla = waitKey(5) & 0xFF
	if tecla == 27:
		break
destroyAllWindows()