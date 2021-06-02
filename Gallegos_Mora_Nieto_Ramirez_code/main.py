#Análisis y procesamiento de imágenes: Proyecto Final Clasificación de fondos oculares para el diagnóstico de enfermedades
#Pamela Ramírez González            #Código: 201822262
#Manuel Gallegos Bustamante         #Código: 201719942
#Sabina Nieto Ramón                 #Código: 201820051
#Nicolás Mora Carrizosa             #Código: 201821509

##Se importan librerías que se utilizarán para el desarrollo del proyecto
from sklearn import ensemble
import argparse
from skimage.feature import hog
import pickle
import skimage.color as color
from tqdm import tqdm
import time
import skimage.transform as transfo
import sklearn.metrics as sk
import cv2
from sklearn import svm
import skimage.io as io
import os
import glob
import numpy as np
from skimage import img_as_float

def charge_imgs(imagen):
    """
    Función para carga de una imaagen con su respectiva anotación
    :param imagen: ruta de la imagen a cargar
    :return: tupla con la carga de la imagen y su anotación en ese orden respectivamente
    """
    matriz_datos = [] #matriz para almacenar datos del archivo cada fila correspondiendo a cada una de las imágenes anotadas
    archivo = open(os.path.join(".","Datos", "Anotaciones","full_df.csv"), mode="r") #lectura del archivo que contiene anotaciones
    titulos = archivo.readline().split(",")
    matriz_datos.append(titulos)
    linea = archivo.readline()
    while len(linea) > 0: #recorrido para lectura de cada una de las imágenes
        linea = linea[1:-2].split(",") # serie de extracciones para obtener los datos en una lista
        linea[1] = int(linea[1])
        linea[-9] = linea[-9][-1:]
        linea[-2] = linea[-2][:2]
        lisa = ""
        for i in range(-9, -1):
            lisa += linea[i]
        lisa = lisa.replace(" ", "")
        del linea[-9:-1]
        unir = np.array([list(lisa)])
        linea[-1:-1] = unir
        linea[-3] = list(linea[-3][2:-2])
        matriz_datos.append(linea)
        linea = archivo.readline() # se procede a evaluar la siguiente línea del archivo
    archivo.close() #se cierra el archivo
    separador = imagen[len(os.path.join(".","Datos")):][0] #separador de rutas dependiente del computador sobre el cual se trabaje
    file_imagen = imagen.split(separador)[-1] #se saca el nombre del archivo de la imagen que entra por parámetro dentro de la ruta
    for i in range(1, len(matriz_datos)): #recorrido para carga y obtención de la anotación de la imagen de interés que entra por la ruta de parámetro
        select_img = matriz_datos[i][-1] #se evalua la imagen en la matriz llenada previamente con la información del archivo con anotaciones
        if file_imagen == select_img: # se corrobora que el archivo de interés corresponde al de la fila en la matriz con anotaciones
            anotacion = matriz_datos[i][-3][0] #se extrae la anotación
            carga = io.imread(imagen) #se carga la imagen
            selected1 = (carga, anotacion) #tupla con información de interés de carga y anotación
            break
    return selected1

def crop_image(image):
    """
    Se realiza el preprocesamiento de la imagen y el cálculo de su descriptor
    :param image: matriz de la imagen a preprocesar
    :return: descriptor de la imagen
    """
    bin_image = image[:, :, 0] # se extrae el canal r de la imagen
    umbral=6 # umbral arbitrario para el cálculo de la máscara binaria
    bin_image=bin_image>umbral # binarización del canal r con el umbral arbitrario
    ancho,largo=len(bin_image),len(bin_image[0]) # dimensiones de la máscara binaria
    recortada=image.copy() # copia de la imagen para recortarla
    recortada[:,:,0],recortada[:,:,1],recortada[:,:,2]=recortada[:,:,0]*bin_image,recortada[:,:,1]*bin_image,recortada[:,:,2]*bin_image # se obtiene el ojo con el fondo negro al multiplicar cada canal por la máscara binaria
    mitad_arriba, mitad_abajo = bin_image[:ancho // 2, :], bin_image[ancho // 2:, :] # variables para hallar la cantidad de ceros a cada uno de los extremos
    sobrantes_arriba, sobrantes_abajo = (len(mitad_arriba) - np.count_nonzero(np.count_nonzero(mitad_arriba, axis=1))), (len(mitad_abajo) - np.count_nonzero(np.count_nonzero(mitad_abajo, axis=1)))
    mitad_izq, mitad_der = bin_image[:, :largo // 2], bin_image[:, largo // 2:]
    sobrantes_izq, sobrantes_der = (len(mitad_izq[0]) - np.count_nonzero(np.count_nonzero(mitad_izq, axis=0))), (len(mitad_der[0]) - np.count_nonzero(np.count_nonzero(mitad_der, axis=0)))
    if sobrantes_der != 0: #condicionales para recorte de la imagen de manera de que se enfoque el ojo
        recortada = recortada[:, :-sobrantes_der]
    if sobrantes_izq != 0:
        recortada = recortada[:, sobrantes_izq:]
    if sobrantes_abajo != 0:
        recortada = recortada[:-sobrantes_abajo, :]
    if sobrantes_arriba != 0:
        recortada = recortada[sobrantes_arriba:, :]

    #PREPROCESAMIENTO HOG
    # 1. tomar la imagen recortada previamente y sacar el 50% del nivel de gris
    #       1.1 Aplicar un filtro Gaussiano con kernel 63x63 y sigma=150
    #       1.2 Restar el filtrado Gaussiano a la imagen original y sumarle el 50% de la intensidad
    # 2. realizar un resize de las imágenes de manera de que queden de 512x512
    # 3. Realizar una apertura de la imagen con un kernel de elipse de 7x7
    k_size = 63
    sigma = 150
    porcentaje_gris = round(256 * 0.5)
    filtradoGauss = cv2.GaussianBlur(recortada, (k_size, k_size), sigma)
    resta_medioloc = (recortada - filtradoGauss) + porcentaje_gris

    preprocesada = transfo.resize(resta_medioloc, (512, 512))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(7, 7))
    preprocesada = cv2.morphologyEx(preprocesada, cv2.MORPH_OPEN, kernel)

    #MODELO HOG
    # Hallar el descriptor de HOG con 30 pixels per cell y una distancia para realizar el histograma de L2-Hys
    pixels_per_cell2=30
    norm_block="L2-Hys"
    resp_descript = hog(preprocesada,block_norm=norm_block, pixels_per_cell=(pixels_per_cell2, pixels_per_cell2))

    return resp_descript

def test_predict(descript):
    """
    se hallan predicciones del modelo
    :param descript: descriptores de las imágenes para realizar predicciones
    :return: predicciones del modelo
    """
    modelo = pickle.load(open(os.path.join(".","modelos", "SVM_HOG_30pxls_L2Hys_linear_apertura50.npy"), 'rb'))
    prediction = modelo.predict(descript)
    return  prediction

def show_metrics():
    """
    Se preprocesan las imágenes de prueba, se hallan sus respectivos descriptores y predicciones haciendo uso de funciones definidas previamente
    Se muestra al usuario resultados del método de clasificación realizado
    """
    inicio = time.time() # se inicia el tiempo del algoritmo
    descripts_test = [] #variables para almacenar descriptores y anotaciones
    anotaciones_test = []
    imagenes = glob.glob(os.path.join(".", "Datos", "Imágenes", "PruebaFINAL", "*.jpg"))   # obtención de rutas de las imágenes de prueba
    nombredescripts_test = "descripts_test_modelofinal.npy"

    for imagen in tqdm(imagenes): # se calculan los descriptores y anotaciones para cada una de las imágenes haciendo uso de las funciones definidas previamente
        descripts_test.append(crop_image(charge_imgs(imagen)[0]))
        anotaciones_test.append(charge_imgs(imagen)[1])

    predicciones = test_predict(descripts_test) # se calculan las predicciones de las imágenes

    conf_mat = sk.confusion_matrix(anotaciones_test, predicciones) # cálculo de métricas del modelo
    precision = sk.precision_score(anotaciones_test, predicciones, average="macro", zero_division=1)
    recall = sk.recall_score(anotaciones_test, predicciones, average="macro", zero_division=1)
    f_score = sk.f1_score(anotaciones_test, predicciones, average="macro")

    fin = time.time() # se finaliza el tiempo que tarda el algoritmo
    demora = fin - inicio
    #  SE MUESTRAN LOS RESULTADOS AL USUARIO
    print("\n---------RESULTADOS---------")
    print(f"\nEl preprocesamiento y clasificación del conjunto de prueba se tardó {demora} sec\n\n"
          f"Métricas generales del modelo:\n"
          f"Precisión: {precision:10} | Cobertura: {recall:10} | F-score: {f_score:10}\n"
          f"Matriz de confusión:\n"
          f"{conf_mat}\n\n"
          f"Reporte de clasificación:\n"
          f"{sk.classification_report(anotaciones_test, predicciones)}")

parser = argparse.ArgumentParser(description='Clasificación de fondos oculares para el diagnóstico de enfermedades') # se inicializa Parser
parser.add_argument('-i', '--imagen', type=str, default=None, help='Nombre de la imagen a evaluar\nEj: 553_left.jpg')
args = parser.parse_args()

if __name__ == '__main__':
    if args.imagen is None: # si no se desea evaluar una imagen en específico se calcula el modelo para todas las imágenes de prueba haciendo uso de la función creada para este fin
        show_metrics()
    else: # si se ingresa la imagen por parámetros

        inicio=time.time() # se inicia el tiempo del algoritmo

        descripts_test = [] #variables para almacenar descriptor y anotación
        anotaciones_test = []
        imagenes = glob.glob(
            os.path.join(".","Datos", "Imágenes", "PruebaFINAL", args.imagen))   # obtención de ruta de la imagen de prueba ingresada por parám
        nombredescripts_test = "descripts_test_modelofinal.npy"

        for imagen in tqdm(imagenes): # se calcula el descriptor y se obtiene la anotación de la imagen ingresada por paáretro haciendo uso de las funciones definidas previamente
            descripts_test.append(crop_image(charge_imgs(imagen)[0]))
            anotaciones_test.append(charge_imgs(imagen)[1])

        predicciones = test_predict(descripts_test) # se predice el modelo haciendo uso de la función creada para esto

        fin=time.time()# se finaliza el tiempo que tarda el algoritmo
        demora=fin-inicio
        #  SE MUESTRAN LOS RESULTADOS AL USUARIO
        print(f"\n---------RESULTADOS para la imagen: {args.imagen}---------")
        print(f"\nPertenece a la clase: {anotaciones_test}\n"
              f"Fue asignada a la clase: {predicciones}\n"
              f"Métrica de clasificación: {int(predicciones==anotaciones_test)}\n\n"
              f"El preprocesamiento y clasificación de la imagen se tardó {demora} sec")
