#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Sabina Nieto Ramón #Código: 201820051
#Nicolás Mora Carrizosa #Código: 201821509
#Análisis y procesamiento de imágenes: Proyecto Final Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
from skimage.filters import threshold_otsu
import nibabel
from scipy.io import loadmat
#import requests
from skimage.color import rgb2gray
import skimage.segmentation as segmenta
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import skimage.io as io
import os
import skimage.morphology as morfo
import glob
##

imagenes=glob.glob(os.path.join("BasedeDatos","preprocessed_images","*.jpg"))
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