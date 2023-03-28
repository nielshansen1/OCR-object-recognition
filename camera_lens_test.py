''' 
Dit script geeft een preview weer van de aangesloten picamera. Tijdens
deze preview kunnen foto's gemaakt worden die de naam 'focustest+nummer.jpg'
krijgen. Het nummer wordt bepaald door de counter (standaard 0). Deze counter
wordt per gemaakte foto met 1 verhoogd. 

Ook zijn de camera instellingen shutter speed en resolution toegevoegd
en aan te passen.

Ten slotte zijn de OCR functionaliteiten ook toegevoegd met Tesseract.
'''

import time
import picamera
import keyboard
import cv2 as cv
import numpy as np
import pytesseract as tess


afbeeldingnaam = 'focustest_maart' +str(counter)+'.png'
camera = picamera.PiCamera()


# functies voor maken v foto 
def take_picture():
	camera.capture(afbeeldingnaam)
	
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
	
#tekstherkenning (met preprocessing)
def tesseract_to_string(image):
	bright = brightness_increase(image, 1)   #helder maken vanwege weinig licht
	noiseless = remove_noise(bright)		 #ruis wegfilteren, schone pixels
	gray = get_grayscale(noiseless)			 #grijsmaken
	sharp = sharpen(gray)
	thresh1 = thresholding(sharp)		 	 #binair thresholden
	text1 = tess.image_to_string(thresh1, config=r'--oem 3 --psm 3')
	#cv.imshow('after preprocessing', thresh1)
	cv.imwrite('bewerkt_'+ afbeeldingnaam, thresh1)
	print(text1)


camera.resolution = (1280, 1024)
camera.shutter_speed = 10000 #gegeven in microseconden
camera.start_preview(fullscreen=False, window = (0,0,720,1600))

while True:	
	keyboard.wait(' ')
	take_picture() 
	img = cv.imread(afbeeldingnaam)
	tesseract_to_string(img)
	print('foto ' + str(counter))
	counter = counter+1
