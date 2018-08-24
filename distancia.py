import cv2
import freenect
import numpy as np
import time

def get_video():
    array,_ = freenect.sync_get_video(0, freenect.VIDEO_RGB)
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#Mostrar la imagen de profundidad
def get_depth():
    array,_ = freenect.sync_get_depth(0,freenect.DEPTH_11BIT)
    array = array.astype(np.uint8)
    return array

#Obtener distancia promedio de toda la imagenm, de la region central y del pixel de en medio
def get_mean_distance_mks():
    array,_ = freenect.sync_get_depth(0,freenect.DEPTH_MM)
    suma = []
    #Distancia del pixel de en medio y promedio, ignorando lo que este marque distancia 0
    bina = array[240][320]/1000
    for x in range (225,255):
        for y in range (305,335):
            suma.append(array[x][y])
    cuadro = np.nanmean(suma)/1000
    dpromedio = np.nanmean(array)/1000
    return dpromedio, cuadro, bina

if __name__ == "__main__":
    while 1:
        distancia, cuad, bina = get_mean_distance_mks()
        BinD = 'Dist a pix central: '+ str(bina)+' Metros'
        StrD = str(cuad)+' Metros'

        #Dibuja cuadro donde se esta tomando la distancia promedio
        fondo= np.ones((200,600,3),np.uint8)
        cv2.putText(fondo,BinD,(10,60), font, 1,(219,169,1),1,cv2.LINE_AA)
        cv2.putText(fondo,'Distancia central promedio',(10,110), font, 1,(0,191,255),1,cv2.LINE_AA)
        cv2.putText(fondo,StrD,(10,160), font, 1,(0,191,255),1,cv2.LINE_AA)

        #Muestra cuadro con valores de distancia del pixel de en medio 
        cv2.imshow('Distancia',fondo)
#Fin del programa
cv2.destroyAllWindows()
