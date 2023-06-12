#! /usr/bin/python3

'''
Dit python script is een GUI applicatie die foto's met een camera kan maken en
deze verplaatsen naar een database om daarvan een Neuraal Netwerk te creeeren.
Daarnaast kan de code dit neuraal netwerk toepassen op foto's en de klasse van het
gefotografeerde object bepalen. 

Ook is het in staat om OCR toe te passen op de foto's. Daarvoor wordt de library
pytesseract gebruikt. Hiervoor wordt eerst de orientatie van de verpakking ingevoerd.

Ten slotte wordt de barcode van de verpakking gelezen door de package pyzbar. Hiermee
wordt de CE barcode van de verpakking op de GUI gezet.
'''

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image as IMAGE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from guizero import App, Picture, Text, PushButton, Window, TextBox, Box
from picamera2.previews import QtGlPreview
from picamera2 import Picamera2, Preview
from pyzbar.pyzbar import decode 
import pytesseract as tess
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import random
import shutil
import glob
import json
import time
import os

# Create the GUI
gui = App(title="Classifier System", height=2000, width=2000, layout="grid")
buttonbox = Box(gui, layout="grid", grid=[0,0])
picturebox = Box(gui, layout="grid", grid=[1,0])
textbox = Box(gui, layout="grid", grid=[1,1], align='left')

# Create the camera with custom settings
camera = Picamera2()
camera.configure(camera.create_still_configuration())
camera.set_controls({"ExposureTime": 5000})
	
# take picture, scan for barcode, apply ocr and use trained model
def take_picture():
	# Start the camera
	camera.start()
	# generate filename for the created picture
	filename = '/home/pi/camera/train/'+ time.strftime("%Y%m%d-%H%M%S")+".png"
	filename_resized = '/home/pi/camera/train/'+ time.strftime("%Y%m%d-%H%M%S")+"_resized.png"
	# create picture with generated filename
	camera.capture_file(filename)
	camera.stop()
	# half the size of the picture 
	with Image.open(filename) as img:
		resized_img = img.resize((img.width//6, img.height//6))
		resized_img.save(filename_resized)
	# update the picture widget with the new image
	picture.value=filename_resized
	# OCR on big picture!
	image = cv.imread(filename)
  #check the image for barcode
	barcodescan(image)
  #check the image for text
	tesseract_to_string(image)
  #use the trained model to classify the package
	usemodel(filename)	
	
# Create panorama picture from multiple saved pictures (NOT CURRENTLY USED IN THIS CODE, 
# WAIT FOR FINAL VERSION OF PAPER!)
def stitch_pictures()
	camera.start()
	dirname=textbox.value
	for i in range(10)
		#maak de path compleet en check of de naam bestaat
		path = os.path.join("/home/pi/camera/modelfotos/", dirname)
		if not os.path.exists(path):
			os.mkdir(path,mode=0o777)
		filename = path + "/" + time.strftime("%Y%m%d-%H%M%S")+".jpg"
		print(filename)
		camera.capture_file(filename)
		cv2.waitKey(300)										

# function to create pictures for the classifier dataset
def picture_to_DS():
	#start de camera
	camera.start()
	#pak de directory naam uit de textbox
	dirname = textbox.value
	#maak de path compleet en check of de naam bestaat
	path = os.path.join("/home/pi/camera/modelfotos/", dirname)
	if not os.path.exists(path):
		os.mkdir(path,mode=0o777)
	filename = path + "/" + time.strftime("%Y%m%d-%H%M%S")+".jpg"
	print(filename)
	camera.capture_file(filename)
	camera.stop()
	
# function that gets the most current trained model file and uses it on the created picture
def usemodel(img_filename):
	# Get Classfile Directory
	class_dir = '/home/pi/camera/classfiles'
	all_classfiles = glob.glob(os.path.join(class_dir, '*.json'))
	newest_classfile = max(all_classfiles, key=os.path.getctime)
	with open(newest_classfile, 'r') as f:
		class_labels= json.load(f)
	# Get Modelfile Directory
	models_dir = '/home/pi/camera/modelfiles'
	# Look for all .h5 files from Modelfile Directory
	all_model_files = glob.glob(os.path.join(models_dir, '*.h5'))
	# Select the newest model based on the hightest number in the filename
	newest_model = max(all_model_files, key=os.path.getctime)
	# Assign newest model to the model variable
	model = tf.keras.models.load_model(newest_model)
	# Use the model on the created image
	testimg = IMAGE.load_img(img_filename, target_size=(200,200))
	x = IMAGE.img_to_array(testimg)
	x = np.expand_dims(x, axis=0)
	prediction = model(x) #model.predict(x) --> model(x) om te checken of tracing errror weggaat
	predicted_class_index = np.argmax(prediction)
	print(predicted_class_index)
	predicted_class_label = class_labels[str(predicted_class_index)]
	print(predicted_class_label)
	class_text.value = predicted_class_label

# function that creates a train/test split in the directories to train a model
def traintestsplit():
	#train & validation directories
	train_dir = "/home/pi/camera/train"
	validation_dir = "/home/pi/camera/validation"

	# list of product names
	koeksoorten = os.listdir(train_dir)

	# Loop through the list of names
	for koeksoort in koeksoorten:
		koeksoort_dir = os.path.join(train_dir, koeksoort)
		validation_koeksoort_dir = os.path.join(validation_dir, koeksoort)
		
		# check if the directory already exists in the validation directory
		if not os.path.exists(validation_koeksoort_dir):
			os.makedirs(validation_koeksoort_dir)
		
		# list of files in the product directory
		bestanden = os.listdir(koeksoort_dir)
		
		# list of files that where alreay in the validation dir
		validation_bestanden = os.listdir(validation_koeksoort_dir)
		
		# calculate the amount of files that need to be moved to validation
		n_validation = int(0.2 * len(bestanden))
		
		# check if there are plenty of files for the validation directory
		if n_validation >= len(bestanden):
			raise ValueError("Niet voldoende bestanden in de directory {} voor een 80/20 split.".format(koeksoort_dir))
		
		# check the amount of files that can be moved
		n_available = len(bestanden) - len(validation_bestanden)
		
		# check if there are plenty of available files for the validation directory
		if n_validation > n_available:
			raise ValueError("Niet voldoende beschikbare bestanden in de directory {} voor een 80/20 split.".format(koeksoort_dir))
		
		# randomly choose files for validation directory
		validation_bestanden += random.sample([bestand for bestand in bestanden if bestand not in validation_bestanden], n_validation)
		
		# move the chosen files to the validation directory
		for bestand in validation_bestanden:
			bestand_src = os.path.join(koeksoort_dir, bestand)
			bestand_dst = os.path.join(validation_koeksoort_dir, bestand)
			shutil.move(bestand_src, bestand_dst)
			print("Bestand {} verplaatst naar {}".format(bestand, validation_koeksoort_dir))
	
# start camera preview
def start_preview():
	camera.start()
	camera.start_preview(QtGlPreview)

# stop camera preview
def stop_preview():
	camera.stop()

# define functions for preprocessing images
def get_grayscale(image):
	return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def canny(image):
	return cv.Canny(image, 100, 200)

def blurring(image):
	return cv.GaussianBlur(image, (7,7), 0)

def thresholding(image):
	return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
	
def remove_noise(image):
	return cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)

def rotate(image):
	height, width = image.shape[:2]
	center = (width/2, height/2)
	rotate_matrix = cv.getRotationMatrix2D(center = center, angle=0, scale=1)
	return cv.warpAffine(src=image, M=rotate_matrix, dsize=(width,height))
	
def ocr_angle_correction(image):
	#preprocessing already needs to be done before calling this function
	lines = cv.HoughLines(image, 1, np.pi/180, 100)
	
	# vind de vaakst voorkomende angle van lijnen in de afbeelding
	angles = []
	for rho, theta in lines[:, 0]:
		angle = np.degrees(theta)
		angles.append(angle)
	dominant_angle = np.median(angles)
	
	height, width = image.shape[:2]
	center = (width/2, height/2)
	rotate_matrix = cv.getRotationMatrix2D(center, dominant_angle, scale=1)
	rotated = cv.warpAffine(image, rotate_matrix, (width,height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
	return rotated
	
def remove_flash(image):
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (5,5))
	return clahe.apply(image)

def brightness_increase(image, minimum_brightness):
	cols, rows = image.shape[:2]
	brightness = np.sum(image)/(255*cols*rows)
	ratio = brightness / minimum_brightness
	if ratio >= 1:
		return image
	
	return cv.convertScaleAbs(image, alpha =1/ratio, beta = 0)
	
def sharpen(image):
	kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
	return cv.filter2D(src=image, ddepth=-1, kernel=kernel)
	
# Function for using OCR on image, first preprocessing the image
def tesseract_to_string(image):
	# Generate filename for ocr image
	filename_OCR = time.strftime("%Y%m%d-%H%M%S")+"_OCR.png"
	filename_OCR_resized = time.strftime("%Y%m%d-%H%M%S")+"_OCR_resized.png"
	# increase brightness
	bright = brightness_increase(image, 1) 
	# filter out noise pixels  
	noiseless = remove_noise(bright)
	# grayscale the picture
	gray = get_grayscale(noiseless)
	# sharpen the text on the picture
	sharp = sharpen(gray)
	# binary threshold to create black/white difference
	thresh1 = thresholding(sharp)	
	#NEW FUNCTION ROTATE THE IMAGE SO TEXT IS HORIZONTAL
	rotated = ocr_angle_correction(thresh1)
	print('image rotated')
	# read text from picture (OCR)	
	text1 = tess.image_to_string(thresh1, config=r'--oem 3 --psm 3')  
	text_no_white=" ".join(text1.split())
	# save OCR image 
	cv.imwrite(filename_OCR , thresh1)
	# write text from image to GUI
	ocrtext.value = text_no_white
	# show the OCR picture next to the normal picture in GUI
	with Image.open(filename_OCR) as img:
		resized_ocr_img = img.resize((img.width//6, img.height//6))
		resized_ocr_img.save(filename_OCR_resized)
		picture_ocr.value=filename_OCR_resized

# function that detects if there barcode in the image
def barcodescan(image):
	barcode = decode(image)
	if(len(barcode) != 0):
		bdata = str(barcode[0].data)
		print(bdata+" gedetecteerde code")
		print(currentbatch+" ingevoerde code")
		if (bdata == currentbatch):
			barcodetext.value = ('barcode komt WEL overeen:\n ingevoerde barcode:'+currentbatch+'\n gedetecteerde barcode: '+bdata)
			barcodetext.text_color = 'green'
		else:
			barcodetext.value = ('barcode komt NIET overeen\n ingevoerde barcode:'+currentbatch+'\n gedetecteerde barcode: '+bdata)
			barcodetext.text_color = 'red'
	else:
		barcodetext.value = ('geen barcode gedetecteerd\n ingevoerde barcode:'+currentbatch+'\n gedetecteerde barcode: geen ')
		barcodetext.text_color = 'black'

# function that can input the current used barcode on the flowpacker to check if barcode is correct
def newbatchname():
	global currentbatch
	currentbatch = ("b'"+textbox2.value+"'")

# function that creates new window for filling in barcode
def newbatch():
	global textbox2
	# Create new window
	window2 = Window(gui, layout = "grid", title="new batch", width = 400, height = 200 )
	text2 = Text(window2, grid=[0,0], text="Product ID:")
	textbox2 = TextBox(window2, grid=[0,1], width=25, height=1)
	newbatchbutton2 = PushButton(window2, grid = [0,2], text="proceed", command=newbatchname)
	
# Starts a learning window where new product can be added to dataset
def learning():
	global textbox
	# Create new window 
	window = Window(gui, layout = "grid", title="create pictures for new product", width=300, height=100 )
	text = Text(window, grid=[0,0], text="Product ID:")
	textbox = TextBox(window, grid=[0,1], width=25, height=1)
	dataset_button = PushButton(window, grid = [0,2], text="take image", command=picture_to_DS)
	
# Starts training the updated dataset	
def train():
	# waardes van 0-1 ipv 0-255
	train = ImageDataGenerator(rescale=1/255)
	validation = ImageDataGenerator(rescale=1/255)

	train_dataset= train.flow_from_directory("/home/pi/camera/train/",
											 target_size=(200, 200),
											 batch_size = 1, #moet waarschijnlijk groter met grotere dataset
											 class_mode = "categorical") 

	validation_dataset= validation.flow_from_directory("/home/pi/camera/validation/",
														target_size=(200, 200),
														batch_size = 1,
														class_mode = "categorical")
	print(validation_dataset.class_indices)												
	model = Sequential()
	
	#convolutielaag met 32 filters van 3,3, ReLU-activatiefuncite en same padding
	model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(200,200,3)))
	
	# max poolinglaag met een poolgrootte van 2x2
	model.add(MaxPooling2D((2,2)))
	
	# flattenlaag om de feature maps om te zetten naar een 1D vector
	model.add(Flatten())
	
	# Volledig verbonden laag met 128 neuronen en ReLU-activatie
	model.add(Dense(128, activation='relu'))
	
	# dropoutlaag met een percentage van 50% om overfitting te verminderen
	model.add(Dropout(0.5))
	
	# uitvoerlaag met softmaxfunctie voor de classificatie van de klassen
	model.add(Dense(6, activation='softmax'))
	
	model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
	
	# PAK HIER DE CLASS NAMEN EN ZET ZE IN EEN DICT																		
	class_labels = validation_dataset.class_indices
	print(class_labels)
	# MODEL FITTEN: TRAIN HET MODEL
	model_fit= model.fit(train_dataset, epochs=3, validation_data=validation_dataset)
	
	#Save Model
	model_name = 'modelfile_1.h5'
	index = 1
	while os.path.exists('/home/pi/camera/modelfiles/'+model_name):
		model_name = f'modelfile_{index}.h5'
		index += 1
	
	model.save('/home/pi/camera/modelfiles/'+ model_name)
	print(f'Model saved as: {model_name}')
	
	#Save classes	
	class_file_name= 'classfile_1.json'
	index=1
	class_labels = validation_dataset.class_indices
	class_labels = dict((v, k) for k, v in class_labels.items())
	while os.path.exists('/home/pi/camera/classfiles/'+class_file_name):
		class_file_name = f'classfile_{index}.json'
		index += 1

	with open('/home/pi/camera/classfiles/'+class_file_name, 'w') as f:																
		json.dump(class_labels, f)
	f.close()
	print(f'classfile created  {class_file_name}')
	
# Create Picture widget
picture = Picture(picturebox, grid=[0,0])
picture_ocr = Picture(picturebox, grid=[1,0])

# Create button to make the pictures
pic_button = PushButton(buttonbox, grid=[0,0], text="take image", command = take_picture)

# Create OCRText widget
ocrtext = Text(textbox, grid=[0,0])

# Create Classifier Text widget
class_text = Text(textbox, grid=[0,1])

# Create Barcode Text widget
barcodetext = Text(textbox, grid=[0,2])

# Create button to add current production product
current_product = PushButton(buttonbox, grid=[0,5], text="new batch", command = newbatch) 

# Create learning button for system to create new product
learn_button = PushButton(buttonbox, grid=[0,1], text="add new product to database", command = learning)

# Create train button to train new model with added data
train_button = PushButton(buttonbox, grid=[0,2], text="train updated model", command=train)

# Create camera preview widget
cam_stop_button = PushButton(buttonbox, grid=[0,3], text="stop preview (not working)", command = traintestsplit)
cam_start_button = PushButton(buttonbox, grid=[0,4], text="start preview (not working)", command = start_preview)

# Run the GUI
gui.display()

