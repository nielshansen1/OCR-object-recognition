#! /usr/bin/python3

import os
import time
import picamera
import cv2 as cv
import numpy as np
import pytesseract as tess
from guizero import App, Picture, Text, PushButton, Window, TextBox
from PIL import Image
import tensorflow as tf

# Create the GUI
gui = App(title="Classifier System", height=2000, width=2000, layout="grid")

# Create the camera with custom settings
camera = picamera.PiCamera()
camera.resolution = (2028, 1520)
camera.shutter_speed = 2500

# Start the preview
camera.start_preview(fullscreen=False, window = (0,0,600,1600))

# functies voor maken v foto 
def take_picture():
	# generate filename for the created picture
	filename = '/home/raspberry/camera/train/'+ time.strftime("%Y%m%d-%H%M%S")+".jpg"
	filename_resized = time.strftime("%Y%m%d-%H%M%S")+"_resized.jpg"
	# create picture with generated filename
	camera.capture(filename)
	# half the size of the picture 
	with Image.open(filename) as img:
		resized_img = img.resize((img.width//3, img.height//3))
		resized_img.save(filename_resized)
	# update the picture widget with the new image
	picture.value=filename_resized
	
	# OCR on big picture!
	image = cv.imread(filename)
	tesseract_to_string(image)

def picture_to_DS():
	dirname = textbox.value
	path = os.path.join("/home/raspberry/camera/train/", dirname)
	if not os.path.exists(path):
		os.mkdir(path,mode=0o777)
	filename = path + "/" + time.strftime("%Y%m%d-%H%M%S")+".jpg"
	print(filename)
	camera.capture(filename)
	print('file saved')
	
def remove_preview():
	camera.stop_preview()

# define functies voor bewerken afbeelding
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
	rotate_matrix = cv2.getRotationMatrix2D(center = center, angle=90, scale=1)
	return cv.warpAffine(src=image, M=rotate_matrix, dsize=(width,height))
	
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
	
# OCR (with preprocessing)
def tesseract_to_string(image):
	# Generate filename for ocr image
	filename_OCR = time.strftime("%Y%m%d-%H%M%S")+"_OCR.jpg"
	filename_OCR_resized = time.strftime("%Y%m%d-%H%M%S")+"_OCR_resized.jpg"
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
	# read text from picture (OCR)	 	 
	text1 = tess.image_to_string(thresh1, config=r'--oem 3 --psm 3')
	text_no_white=" ".join(text1.split())
	# save OCR image 
	cv.imwrite(filename_OCR , thresh1)
	# print text from image in command window
	print(text1)
	# write text from image to GUI
	ocrtext.value = text_no_white
	# show the OCR picture next to the normal picture in GUI
	with Image.open(filename_OCR) as img:
		resized_ocr_img = img.resize((img.width//3, img.height//3))
		resized_ocr_img.save(filename_OCR_resized)
	picture_ocr.value=filename_OCR_resized
	
# Starts a learning window where new product can be added to dataset
def learning():
	global textbox
	# Create new window to work in 
	window = Window(gui, layout = "grid", title="create pictures for new product", width=300, height=100 )
	text = Text(window, grid=[0,0], text="Product ID:")
	textbox = TextBox(window, grid=[0,1], width=25, height=1)
	dataset_button = PushButton(window, grid = [0,2], text="take image", command=picture_to_DS)

# Starts training the updated dataset	
def train():
	#data = DataLoader.from_folder("/home/raspberry/camera/train")
	#model=image_classifier.create(train_data)
	#model.export(export_dir="/home/raspberry/camera")
	return 0 
	
# Create Picture widget
picture = Picture(gui, grid=[2,0])
picture_ocr = Picture(gui, grid=[3,0])

# Create button to make the pictures
pic_button = PushButton(gui, grid=[0,0],text="take image",command = take_picture)

# Create Text widget
ocrtext = Text(gui, grid=[1,0])

# Create learning button for system to create new product
learn_button = PushButton(gui, grid=[0,2], text="add new product to database", command = learning)

# Create train button to train new model with added data
train_button = PushButton(gui, grid=[0,3], text="train updated model", command=train)

# Create camera preview widget
cam_button = PushButton(gui, grid=[0,1], text="remove preview", command = remove_preview)

# Run the GUI
gui.display()



