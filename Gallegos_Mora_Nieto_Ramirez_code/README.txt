Análisis y procesamiento de imágenes: Proyecto Final Clasificación de fondos oculares para el diagnóstico de enfermedades
Pamela Ramírez González            Código: 201822262
Manuel Gallegos Bustamante         Código: 201719942
Sabina Nieto Ramón                 Código: 201820051
Nicolás Mora Carrizosa             Código: 201821509

CONTEXTO: 
Desarrollo de un algoritmo para la clasificación de imágenes de fondos oculares para el diagnóstico de enfermedades. Las clases de clasificación corresponden a la clase control referente a la normalidad (N) y 3 patologías (G: glucoma, C: cataratas y M: miopía). De esta forma, se realiza el preprocesamiento y clasificación de las imágenes de fondo de ojo haciendo uso de 'Machine Learning'. 

LIBRERÍAS:
Las librerías utilizadas para el desarrollo del algoritmo en el archivo 'main.py' son:

	- scikit-image
	- sklearn 
	- argparse
	- pickle
	- tqdm 
	- time
	- OpenCV
	- glob
	- numpy

* De ser necesario las librerías se deben importar por medio del comando 'pip install'

ARCHIVOS: 
> Gallegos_Mora_Nieto_Ramirez_code 
	> Datos 'carpeta que contiene imágenes de evaluación y anotaciones
		> Anotaciones 'carpeta con archivo de anotaciones 
			- full_df.csv 'archivo con anotaciones de las imágenes
		> Imágenes
			> PruebaFINAL 'carpeta con las imágenes de prueba
	> libs 'carpeta vacía ya que no se requiere para guardar otras funciones
	> modelos 'carpeta que contiene el modelo de clasificación desarrollado 
		- SVM_HOG_30pxls_L2Hys_linear_apertura50.npy 
	- main.py 'archivo que contiene el algoritmo para la clasificación de las imágenes de prueba
	- copy.txt 'copia del archivo main.py y códigos desarrollados para la sección de experimentación (entrenamiento y validación)
	- README.txt 'instrucciones 

INSTRUCCIONES:
1. De ser necesario importe las librerías utilizadas en el algoritmo
2. Corra el archivo 'main.py' 
	2.1. Agregue el argumento -i si desea evaluar solo una imagen
		ej. -i 553_left.jpg
	2.2. Omita el argumento -i para evaluar todas las imágenes de prueba 
* Los nombres de las imágenes de prueba se pueden obtener de los archivos en la carpeta  Gallegos_Mora_Nieto_Ramirez_code/Datos/Imágenes 
	
