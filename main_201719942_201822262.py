#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Sabina Nieto Ramón #Código: 201820051
#Nicolás Mora Carrizosa #Código: 201821509
#Análisis y procesamiento de imágenes: Proyecto Final Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import nibabel
from scipy.io import loadmat
from functools import partial
from sklearn import ensemble
from skimage.feature import hog
#import requests
import pickle
from scipy.ndimage import minimum_filter
from skimage.color import rgb2gray
import skimage.color as color
from tqdm import tqdm
import skimage.transform as transfo
import skimage.segmentation as segmenta
import sklearn.metrics as sk
import numpy as np
import cv2
from  PIL import Image
from sklearn import svm
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.io as io
from skimage import exposure
import os
import skimage.morphology as morfo
import glob
##

def charge_imgs(imagen):
    #imagenes = glob.glob(os.path.join("BasedeDatos", "Entrenamiento", "Entrena1", "*.jpg"))
    matriz_datos = []
    archivo = open(os.path.join("BasedeDatos", "full_df.csv"), mode="r")
    titulos = archivo.readline().split(",")
    matriz_datos.append(titulos)
    linea = archivo.readline()
    while len(linea) > 0:
        linea = linea[1:-2].split(",")
        # linea[0]=int(linea[0])
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
        # print(linea)
        matriz_datos.append(linea)
        linea = archivo.readline()
    archivo.close()
    #selected1 = #{}
    #selected2 = {}
    #cont = 1

    separador = imagen[len("BasedeDatos"):][0]
    file_imagen = imagen.split(separador)[-1]

    for i in range(1, len(matriz_datos)):
        select_img = matriz_datos[i][-1]

        if file_imagen == select_img:
            # print(cont)
            anotacion = matriz_datos[i][-3][0]
            # print(anotacion)
            carga = io.imread(imagen)
            selected1 = (carga, anotacion)
            if anotacion!="N" and anotacion!="C" and anotacion!="D" and anotacion!="G":
                print(anotacion)
                print(select_img)
                print(file_imagen)
            # print(select_img)
            break

    """print(select_img)
    print(file_imagen)"""

    """for imagen in imagenes:
        separador = imagen[len("BasedeDatos"):][0]
        file_imagen = imagen.split(separador)[-1]
        for i in range(1, len(matriz_datos)):
            select_img = matriz_datos[i][-1]
            if cont < len(imagenes) / 2:
                if file_imagen == select_img:
                    #print(cont)
                    anotacion = matriz_datos[i][-3][0]
                    # print(anotacion)
                    carga = io.imread(imagen)
                    selected1[cont] = [carga, anotacion]
                    # print(select_img)
                    cont += 1
            else:
                if file_imagen == select_img:
                    anotacion = matriz_datos[i][-3][0]
                    # print(anotacion)
                    carga = io.imread(imagen)
                    selected2[cont] = [carga, anotacion]
                    # print(select_img)
                    cont += 1"""
    #print(len(selected1))
    return selected1#,selected2

##
def crop_image(image,descript):
    """plt.figure() #original
    plt.imshow(image, cmap="gray")
    plt.show()"""
    bin_image = image[:, :, 0]
    #bin_image = rgb2gray(image)
    #umbral = threshold_otsu(bin_image) #TODO PREGUNTAR SOBRE UMBRAL ARBITARIO, PERCENTIL O...
    #print(umbral)
    umbral=6
    bin_image=bin_image>umbral
    """plt.figure()
    plt.imshow(bin_image,cmap="gray") #círculo bin
    plt.show()"""
    ancho,largo=len(bin_image),len(bin_image[0])
    recortada=image.copy()
    recortada[:,:,0],recortada[:,:,1],recortada[:,:,2]=recortada[:,:,0]*bin_image,recortada[:,:,1]*bin_image,recortada[:,:,2]*bin_image
    mitad_arriba, mitad_abajo = bin_image[:ancho // 2, :], bin_image[ancho // 2:, :]
    sobrantes_arriba, sobrantes_abajo = (len(mitad_arriba) - np.count_nonzero(np.count_nonzero(mitad_arriba, axis=1))), (len(mitad_abajo) - np.count_nonzero(np.count_nonzero(mitad_abajo, axis=1)))
    mitad_izq, mitad_der = bin_image[:, :largo // 2], bin_image[:, largo // 2:]
    sobrantes_izq, sobrantes_der = (len(mitad_izq[0]) - np.count_nonzero(np.count_nonzero(mitad_izq, axis=0))), (len(mitad_der[0]) - np.count_nonzero(np.count_nonzero(mitad_der, axis=0)))
    if sobrantes_der!=0 and sobrantes_izq!=0 and sobrantes_arriba!=0 and sobrantes_abajo!=0:
        recortada = recortada[sobrantes_arriba:-sobrantes_abajo, sobrantes_izq:-sobrantes_der]
    elif sobrantes_der==0 and sobrantes_izq==0:
        recortada = recortada[sobrantes_arriba:-sobrantes_abajo, :]
    elif sobrantes_abajo == 0 and sobrantes_arriba == 0:
        recortada = recortada[:, sobrantes_izq:-sobrantes_der]
    ancho_recorte,largo_recorte=len(recortada),len(recortada[0])
    # cantidad_ancho=ancho-np.count_nonzero(sobrantes_ancho)     # cantidad_largo=(largo-(ancho-cantidad_ancho))//2    # recortada=recortada.crop((cantidad_largo, cantidad_ancho/2, cantidad_largo, cantidad_ancho/2))
    """if ancho_recorte<largo_recorte:
        dif=(largo_recorte-ancho_recorte)//2
        recortada=recortada[:,dif:-dif]
    elif largo_recorte<ancho_recorte:
        dif = (ancho_recorte - largo_recorte) // 2
        recortada = recortada[dif:-dif,:]"""
    """plt.figure()
    plt.imshow(recortada, cmap="gray")
    plt.show()"""
    k_size=63
    sigma=150
    porcentaje_gris=round(256*0.7)
    filtradoGauss=cv2.GaussianBlur(recortada,(k_size,k_size),sigma)
    resta_medioloc=(recortada-filtradoGauss)+porcentaje_gris
    recortada = transfo.resize(recortada, (512, 512))
    """plt.figure()
    plt.imshow(resta_medioloc, cmap="gray")
    plt.show()"""
    preprocesada= transfo.resize(resta_medioloc, (512, 512))
    """plt.figure()
    plt.imshow(preprocesada, cmap="gray")
    plt.show()"""
    k_size = 7
    sigma = 20
    #preprocesada = cv2.GaussianBlur(preprocesada, (k_size, k_size), sigma)
    #preprocesada=cv2.medianBlur(preprocesada, 5)

    preprocesada = exposure.adjust_gamma(preprocesada, gamma=5)

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(7, 7))
    #preprocesada= cv2.morphologyEx(preprocesada, cv2.MORPH_OPEN, kernel)
    #preprocesada = exposure.adjust_gamma(preprocesada, gamma=2)
    """plt.figure()
    plt.imshow(preprocesada, cmap="gray")
    plt.show()"""

    """plt.figure()
    plt.imshow(preprocesada, cmap="gray")
    plt.show()"""

    if descript=="COLOR":
        preprocesada = color.rgb2lab(preprocesada)
        """plt.figure()
        plt.imshow(preprocesada, cmap="gray")
        plt.show()"""
        espacio_canal1 = preprocesada[:, :, 0]  # se extraen canales de la imagen en el espacio de color indicado por parámetro
        espacio_canal2 = preprocesada[:, :, 1]
        espacio_canal3 = preprocesada[:, :, 2]
        frec_canal1 = np.histogram(espacio_canal1.flatten(), bins=np.arange(0, 256,   1))  # a cada canal se le realiza un .flantten para trabajan con su vector al cual se le van a sacar las frecuencias de valores entre 0 y 255 con np.histogram
        frec_canal2 = np.histogram(espacio_canal2.flatten(), bins=np.arange(0, 256, 1))
        frec_canal3 = np.histogram(espacio_canal3.flatten(), bins=np.arange(0, 256, 1))
        hist_prev = np.concatenate([frec_canal1[0], frec_canal2[0], frec_canal3[0]],      axis=None)  # se concatenan los arreglos de las frecuencias de cada canal con .concatenate
        resp_descript = hist_prev / np.sum(hist_prev)
    elif descript=="HOG":
        pixels_per_cell2=30
        norm_block="L2-Hys" #{‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
        resp_descript = hog(preprocesada,block_norm=norm_block, pixels_per_cell=(pixels_per_cell2, pixels_per_cell2))

    """plt.figure()
    plt.imshow(recortada, cmap="gray")
    plt.show()"""
    """plt.show()
    plt.figure()
    plt.imshow(preprocesada, cmap="gray")
    plt.show()"""
    return resp_descript
    #TODO PREGUNTAR clasificación supervisada :
    # teniendo en cuenta cómo se diagnostican las patologías que queremos (simplificación de clases a clasificar confirmar)
    # cómo se incorporan las etiquetas?
    # Un solo método o varios? Un descrip, varios? Cómo se haría?
    # *en caso de necesitar entrenamiento como en lab problemas de error de memoria
    #TODO excepciones del preprocesamiento que tenemos: imágenes demasiado recortadas y problemas con el umbral (pregunta de arriba)
##
def matrix_descript():
    m=charge_imgs()
    descripts=np.array([])
    for i in m:
        np.append(descripts,m[i])


##
#cont=0
descripts=[]
anotaciones=[]
imagenes = glob.glob(os.path.join("BasedeDatos", "Entrenamiento", "EvaluacionFIN", "*.jpg"))
for imagen in tqdm(imagenes):
    descripts.append(crop_image(charge_imgs(imagen)[0],descript="HOG"))
    anotaciones.append(charge_imgs(imagen)[1])
    """if cont==3:
        break
    cont+=1"""

kernel_svm='linear'  #{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
entrenamiento_SVM = svm.SVC(kernel=kernel_svm).fit(descripts, anotaciones)
pickle.dump(entrenamiento_SVM, open("SVM_HOG_30pxls_L2Hys_linear_gamma5.npy", 'wb'))
print("calculó modelo")

descripts_valida=[]
anotaciones_valida=[]
imagenes = glob.glob(os.path.join("BasedeDatos", "Validacion", "Proyecto_Validacion", "*.jpg"))
for imagen in tqdm(imagenes):
    descripts_valida.append(crop_image(charge_imgs(imagen)[0],descript="HOG"))
    anotaciones_valida.append(charge_imgs(imagen)[1])

modelo = pickle.load(open("SVM_HOG_30pxls_L2Hys_linear_gamma5.npy", 'rb'))
predicciones = modelo.predict(descripts_valida)
conf_mat = sk.confusion_matrix(anotaciones_valida, predicciones)
precision = sk.precision_score(anotaciones_valida, predicciones, average="macro", zero_division=1)
recall = sk.recall_score(anotaciones_valida, predicciones, average="macro", zero_division=1)
f_score = sk.f1_score(anotaciones_valida, predicciones, average="macro")
print(conf_mat)
print(precision)
print(recall)
print(f_score)
##
cargas=charge_imgs()[0]
##
cargas2=charge_imgs()[1]
##
print(len(cargas))
##

select_rand=np.random.randint(1,len(cargas)-1,6)
print(select_rand)
##
cont=0
for i in select_rand:
    #print(cargas)
    crop_image(cargas[i][0])
    cont+=1
    if cont>5:
        break
##


imagenes=glob.glob(os.path.join("BasedeDatos","Entrenamiento","Entrena2","*.jpg"))
#imagenes=glob.glob(os.path.join("BasedeDatos","preprocessed_images","*.jpg"))
matriz_datos=[]
archivo=open(os.path.join("BasedeDatos","full_df.csv"),mode="r")
titulos=archivo.readline().split(",")
matriz_datos.append(titulos)
linea=archivo.readline()
while len(linea)>0:
    linea = linea[1:-2].split(",")
    #linea[0]=int(linea[0])
    linea[1] = int(linea[1])
    linea[-9] = linea[-9][-1:]
    linea[-2]=linea[-2][:2]
    lisa=""
    for i in range(-9,-1):
        lisa+=linea[i]
    lisa=lisa.replace(" ","")
    del linea[-9:-1]
    unir=np.array([list(lisa)])
    linea[-1:-1]=unir
    linea[-3] = list(linea[-3][2:-2])
    #print(linea)
    matriz_datos.append(linea)
    linea=archivo.readline()
archivo.close()
select_rand=np.random.randint(1,len(matriz_datos)-1,4)
#print(select_rand)
selected1={}
selected2={}
cont=1
for imagen in imagenes:
    separador = imagen[len("BasedeDatos"):][0]
    file_imagen = imagen.split(separador)[-1]
    for i in range(1,len(matriz_datos)):
        select_img=matriz_datos[i][-1]
        if cont<len(imagenes)/2:
            if file_imagen==select_img:
                anotacion=matriz_datos[i][-3][0]
                #print(anotacion)
                carga=io.imread(imagen)
                selected1[cont] = [carga, anotacion]
                #print(select_img)
                cont += 1
        else:
            if file_imagen==select_img:
                anotacion=matriz_datos[i][-3][0]
                #print(anotacion)
                carga=io.imread(imagen)
                selected2[cont] = [carga, anotacion]
                #print(select_img)
                cont += 1
        #if file_imagen=="300_left.jpg":
print(len(selected1),len(selected2))
##
circulo_temp=io.imread(os.path.join('BasedeDatos',"preprocessed_images", "300_left.jpg"))
filtrado = cv2.medianBlur(circulo_temp, 3)
grises = rgb2gray(filtrado)
L = np.amax(grises)
neg = grises.copy()
for i in range(len(grises)):
    for j in range(len(grises[0])):
        formula = (L) - grises[i][j]
        neg[i][j] = formula
gamma = exposure.adjust_gamma(neg, gamma=2)
# print(len(gamma))
umbral = threshold_otsu(gamma)
circulo_temp = gamma < umbral
print(len(selected1),len(selected2))
##
select_rand=np.random.randint(1,len(matriz_datos)-1,4)
print(select_rand)
selected={}
cont=1
for i in select_rand:
    select_img=matriz_datos[i][-1]
    for imagen in imagenes:
        separador=imagen[len("BasedeDatos"):][0]
        file_imagen=imagen.split(separador)[-1]
        #print(imagen)
        #print(select_img)
        if file_imagen==select_img:
            #selected[cont]=[imagen,matriz_datos[i][-3][0]]
            anotacion=matriz_datos[i][-3][0]
            print(anotacion)
            carga=io.imread(imagen)
            selected[cont] = [carga, anotacion]
            #carga=nibabel.load(imagen)
            #print(carga)
            print(select_img)
            #print(imagen)
            #print("__")
            cont += 1
        elif file_imagen=="300_left.jpg":
            circulo_temp=io.imread(imagen)
            filtrado = cv2.medianBlur(circulo_temp, 3)
            grises = rgb2gray(filtrado)
            L = np.amax(grises)
            neg = grises.copy()
            for i in range(len(grises)):
                for j in range(len(grises[0])):
                    formula = (L) - grises[i][j]
                    neg[i][j] = formula
                    """if formula<0:
                        neg[i][j]=0
                    else:
                        neg[i][j] = formula"""
    # print(neg)
    # gamma=neg^(1/2)#np.power(neg,1/2)#.clip(0,255).astype(np.uint8)

    gamma = exposure.adjust_gamma(neg, gamma=2)
    # print(len(gamma))
    umbral = threshold_otsu(gamma)
    circulo_temp = gamma < umbral


plt.figure()
plt.subplot(4,2,1)
plt.imshow(selected[1][0])
plt.axis("off")
plt.subplot(4,2,2)
plt.text(0.5, 0.5, selected[1][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,3)
plt.imshow(selected[2][0])
plt.axis("off")
plt.subplot(4,2,4)
plt.text(0.5, 0.5, selected[2][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,5)
plt.imshow(selected[3][0])
plt.axis("off")
plt.subplot(4,2,6)
plt.text(0.5, 0.5, selected[3][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,7)
plt.imshow(selected[4][0])
plt.axis("off")
plt.subplot(4,2,8)
plt.text(0.5, 0.5, selected[4][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.show()
##
#print(selected[4][0])

def idea_dif(image):
    grises=rgb2gray(image)
    #print(np.amax(grises),np.amin(grises))
    unicos=np.unique(grises)
    unicos=unicos[unicos!=0]
    prom=np.mean(unicos)
    maximo=np.amax(unicos)
    dif=maximo-prom
    umbral=0.30
    print(prom,maximo,dif)
    if dif>umbral:
        print("No hay carataras")
    else:
        print("Cataratas")
def idea_bin(image):
    #bin_image=rgb2gray(image)
    imagen_en_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bin_image = image[:,:,1]
    espacio_b = imagen_en_Lab[:, :, 2]
    umbral = threshold_otsu(bin_image)
    umbral=threshold_otsu(espacio_b)
    bin_image=bin_image>umbral
    #bin_image = espacio_b < umbral
    #dilatacion = morfo.dilation(grises)
    #erosion = morfo.erosion(grises)

    return bin_image#dilatacion-erosion#bin_image
def idea_water(image):
    grises = rgb2gray(image)
    #bin_image=rgb2gray(image)
    imagen_en_Lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    bin_image = image[:,:,1]
    espacio_b = imagen_en_Lab[:, :, 2]
    umbral = threshold_otsu(bin_image)
    umbral=threshold_otsu(espacio_b)
    marks=bin_image>umbral
    marks=morfo.erosion(marks)

    #bin_image = espacio_b < umbral
    #marks = morfo.h_minima(grises, 40)
    dilatacion = morfo.dilation(grises)
    erosion = morfo.erosion(grises)
    grad=dilatacion-erosion
    bin_image=segmenta.watershed(grad, markers=marks, watershed_line=True)
    return bin_image#grad#dilatacion-erosion#bin_image

#print(preprocesamiento_binario(selected[4][0]))
plt.figure()
plt.subplot(4,2,1)
plt.imshow(idea_water(selected[1][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,2)
plt.text(0.5, 0.5, selected[1][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,3)
plt.imshow(idea_water(selected[2][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,4)
plt.text(0.5, 0.5, selected[2][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,5)
plt.imshow(idea_water(selected[3][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,6)
plt.text(0.5, 0.5,selected[3][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,7)
plt.imshow(idea_water(selected[4][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,8)
plt.text(0.5, 0.5, selected[4][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.show()

##
"""plt.subplot(1,2,1)
plt.imshow(selected[3][0])
plt.axis("off")
plt.subplot(1,2,2)
plt.text(0.5, 0.5, selected[3][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.imshow(preprocesamiento_binario(selected[3][0]),cmap="gray")"""
plt.figure()
plt.subplot(4,2,1)
plt.imshow(preprocesamiento_binario(selected[1][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,2)
plt.text(0.5, 0.5, selected[1][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,3)
plt.imshow(preprocesamiento_binario(selected[2][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,4)
plt.text(0.5, 0.5, selected[2][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,5)
plt.imshow(preprocesamiento_binario(selected[3][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,6)
plt.text(0.5, 0.5,selected[3][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.subplot(4,2,7)
plt.imshow(preprocesamiento_binario(selected[4][0]),cmap="gray")
plt.axis("off")
plt.subplot(4,2,8)
plt.text(0.5, 0.5, selected[4][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.axis("off")
plt.show()
##
circ=np.reshape( io.imread(os.path.join('BasedeDatos', 'circ.png')),512) # se importa la imagen de prueba
plt.imshow(circ)
##print(len(circ),len(circ[0]))
def idea_miopia_neg(image1):
    #image=np.resize(image,(512,512))
    #image = image1.resize((512, 512))
    bin_image = cv2.resize(image1[:, :, 1], dsize=(512,512))
    umbral = threshold_otsu(bin_image)
    filtrado = cv2.medianBlur(bin_image, 3)
    grises = rgb2gray(filtrado)
    L = np.amax(grises)
    neg = grises.copy()
    for i in range(len(grises)):
        for j in range(len(grises[0])):
            formula = (L) - grises[i][j]
            neg[i][j] = formula
    gamma = exposure.adjust_gamma(neg, gamma=2)
    # print(len(gamma))
    umbral = threshold_otsu(gamma)
    circulo_temp = gamma < umbral



    #bin_image=cv2.resize(bin_image>umbral, dsize=(512,512))
    image=cv2.resize(image1, dsize=(512,512))
    filtrado = cv2.medianBlur(image, 3)
    grises = rgb2gray(filtrado)
    L=np.amax(grises)
    neg=grises.copy()
    for i in range(len(grises)):
        for j in range(len(grises[0])):
            formula=(L)-grises[i][j]
            neg[i][j] = formula
            """if formula<0:
                neg[i][j]=0
            else:
                neg[i][j] = formula"""
    #print(neg)
    #gamma=neg^(1/2)#np.power(neg,1/2)#.clip(0,255).astype(np.uint8)

    gamma=exposure.adjust_gamma(neg,gamma=2)
    #print(len(gamma))
    gamma=gamma*circulo_temp

    #gamma #= gamma * bin_image
    umbral = threshold_otsu(gamma)
    vasos = gamma > umbral
    recorte=np.delete(gamma,slice(75),axis=1)
    recorte = np.delete(recorte, slice(75), axis=0)
    recorte = np.delete(recorte, slice(len(recorte)-75,len(recorte)), axis=0)
    recorte = np.delete(recorte, slice(len(recorte[0]) - 75, len(recorte[0])), axis=1)
    #recorte = np.delete(recorte, slice(5), axis=1)
    #gamma=np.where(gamma==np.amax(gamma), 0, gamma)
    #gamma = np.where(gamma == np.amax(gamma), 0, gamma)
    #neg_color=cv2.cvtColor(gamma, cv2.COLOR_GRAY2RGB)
    #I_R=neg_color[:,:,0]
    #I_G=neg_color[:,:,1]
    #I_Q=I_R/(I_G+1)
    #umbral = threshold_otsu(I_Q)
    #vasos = I_Q < umbral
    minimo=minimum_filter(gamma,size=(len(gamma),len(gamma[0])))
    top_hat=morfo.white_tophat(gamma,morfo.square(7))
    #umbral=threshold_otsu(recorte)
   # vasos=recorte>umbral
    dilatacion = morfo.dilation(grises)
    erosion = morfo.erosion(grises)
    grad = dilatacion - erosion
    #bin_image = segmenta.watershed(grad, markers=marks, watershed_line=True)
    if np.count_nonzero(vasos) > 0.90 * (len(vasos)*len(vasos[0])):
        prediccion = "no hay"
    else:
        prediccion = "M"
    return vasos, prediccion #recorte#neg#grad#gamma#
##
circulo=idea_miopia_neg(selected[3][0])
print(np.set_printoptions())
"""#circulo_temp=
for c in range(len(circulo)):
    for cj in range(len(circulo[0])):
        circulo_temp[c][cj]=circulo[c][cj]"""
##
plt.figure()
plt.subplot(4,2,1)
plt.imshow(idea_miopia_neg(selected1[11][0])[0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,2)
#plt.text(0.5, 0.5, selected1[11][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.imshow(selected1[11][0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,3)
plt.imshow(idea_miopia_neg(selected1[12][0])[0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,4)
#plt.text(0.5, 0.5, selected1[12][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.imshow(selected1[12][0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,5)
plt.imshow(idea_miopia_neg(selected1[14][0])[0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,6)
#plt.text(0.5, 0.5,selected1[13][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.imshow(selected1[14][0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,7)
plt.imshow(idea_miopia_neg(selected1[18][0])[0],cmap="gray")
plt.axis("off")
plt.subplot(4,2,8)
#plt.text(0.5, 0.5, selected1[18][1], horizontalalignment="center",verticalalignment="center",fontsize="xx-large",fontweight="semibold")
plt.imshow(selected1[18][0],cmap="gray")
plt.axis("off")
plt.show()

##
TP = 0
TN = 0
FP = 0
FN = 0
##
for i in selected1:
    img_etrenamiento = selected1[i][0]
    img_anotacion = selected1[i][1]
    prediccion_entrena = idea_miopia_neg(img_etrenamiento)[1]
    if prediccion_entrena == "M" and img_anotacion == "M":
        TP += 1
        #plt.imshow(idea_miopia_neg(img_etrenamiento)[0])
        #break
    elif prediccion_entrena == "M" and img_anotacion != "M":
        FP += 1
    elif prediccion_entrena != "M" and img_anotacion == "M":
        FN += 1
    else:
        TN += 1


for i in selected2:
    img_etrenamiento = selected2[i][0]
    img_anotacion = selected2[i][1]
    prediccion_entrena = idea_miopia_neg(img_etrenamiento)[1]
    if prediccion_entrena == "M" and img_anotacion == "M":
        TP += 1
    elif prediccion_entrena == "M" and img_anotacion != "M":
        FP += 1
    elif prediccion_entrena != "M" and img_anotacion == "M":
        FN += 1
    else:
        TN += 1
##
prec=TP/(TP+FP)
cob=TP/(TP+FN)
print(prec,cob)
