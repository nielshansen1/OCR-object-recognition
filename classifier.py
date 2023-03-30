#! /usr/bin/python3

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image as IMAGE
import os
import time
import cv2 as cv
import numpy as np
import pytesseract as tess
from picamera2 import Picamera2, Preview
from guizero import App, Picture, Text, PushButton, Window, TextBox
from PIL import Image
import tensorflow as tf

# Create the GUI
gui = App(title="Classifier System", height=2000, width=2000, layout="grid")

# Create the camera with custom settings
camera = Picamera2()
camera.configure(camera.create_preview_configuration())
camera.set_controls({"ExposureTime": 2500})

# Start the camera
camera.start()
	
# take picture & apply ocr
def take_picture():
	# generate filename for the created picture
	filename = '/home/pi/camera/train/'+ time.strftime("%Y%m%d-%H%M%S")+".png"
	filename_resized = '/home/pi/camera/train/'+ time.strftime("%Y%m%d-%H%M%S")+"_resized.png"
	# create picture with generated filename
	camera.capture_file(filename)
	# half the size of the picture 
	with Image.open(filename) as img:
		resized_img = img.resize((img.width//3, img.height//3))
		resized_img.save(filename_resized)
	# update the picture widget with the new image
	picture.value=filename	
	# OCR on big picture!
	image = cv.imread(filename)
	tesseract_to_string(image)
	usemodel(filename)

def picture_to_DS():
	dirname = textbox.value
	path = os.path.join("/home/pi/camera/train/", dirname)
	if not os.path.exists(path):
		os.mkdir(path,mode=0o777)
	filename = path + "/" + time.strftime("%Y%m%d-%H%M%S")+".jpg"
	print(filename)
	camera.capture_file(filename)
	
def usemodel(img_filename):
	testimg = IMAGE.load_img(img_filename, target_size=(200,200))
	x = IMAGE.img_to_array(testimg)
	x = np.expand_dims(x, axis=0)
	val = model.predict(x)
	print(val)
'''	if val > 0.5:
		print("doos")
		class_text.value = "doos"+str(val)
	else:
		print("boek")
		class_text.value = "boek"+str(val)
'''	
	
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
		picture_ocr.value=filename_OCR
	
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
	global model
	# waardes van 0-1 ipv 0-255
	train = ImageDataGenerator(rescale=1/255)
	validation = ImageDataGenerator(rescale=1/255)

	train_dataset= train.flow_from_directory("/home/pi/camera/train/",
											 target_size=(200, 200),
											 batch_size = 1,
											 class_mode = "binary") #vanwege 2 classes: voor & achterkant

	validation_dataset= validation.flow_from_directory("/home/pi/camera/validation/",
														target_size=(200, 200),
														batch_size = 3,
														class_mode = "binary") #vanwege 2 classes: voor & achterkant
														
	model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(200,200,3)),
										tf.keras.layers.MaxPool2D(2,2),
										#
										tf.keras.layers.Conv2D(32,(3,3),activation='relu', input_shape=(200,200,3)),
										tf.keras.layers.MaxPool2D(2,2),
										#
										tf.keras.layers.Conv2D(64,(3,3),activation='relu', input_shape=(200,200,3)),
										tf.keras.layers.MaxPool2D(2,2),
										##
										tf.keras.layers.Flatten(),
										##
										tf.keras.layers.Dense(512,activation="relu"),
										##
										tf.keras.layers.Dense(1,activation="sigmoid")
										])
										
	model.compile(loss='binary_crossentropy',
				  optimizer = RMSprop(learning_rate=0.001),
				  metrics = ['accuracy'])

	print(validation_dataset.class_indices)

	model_fit= model.fit(train_dataset, steps_per_epoch = 10, epochs=40, validation_data=validation_dataset)
	return 0 
	
# Create Picture widget
picture = Picture(gui, grid=[2,0])
picture_ocr = Picture(gui, grid=[3,0])

# Create button to make the pictures
pic_button = PushButton(gui, grid=[0,0], text="take image", command = take_picture)

# Create OCRText widget
ocrtext = Text(gui, grid=[1,0])

# Create Classifier Text widget
class_text = Text(gui, grid=[1,1])

# Create learning button for system to create new product
learn_button = PushButton(gui, grid=[0,2], text="add new product to database", command = learning)

# Create train button to train new model with added data
train_button = PushButton(gui, grid=[0,3], text="train updated model", command=train)

# Create camera preview widget
cam_button = PushButton(gui, grid=[0,1], text="remove preview", command = remove_preview)

# Run the GUI
gui.display()

