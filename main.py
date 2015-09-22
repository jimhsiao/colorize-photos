# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv

import numpy as np
import cv2 as cv2
import math
import os



# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def NCC(a, b):

	flattened_one = np.ravel(a)
	flattened_two = np.ravel(b)
	flattened_one = flattened_one / np.linalg.norm(flattened_one)
	flattened_two = flattened_two / np.linalg.norm(flattened_two)

	res = np.correlate(flattened_one, flattened_two)
	return res;

# for [-15:15] score each
def findDisplacement(firstPlate, secondPlate):
	temp = firstPlate.shape
	x_dimensions = temp[0]
	y_dimensions = temp[1]
	x_toCrop = x_dimensions//10
	x_cropUntil = x_dimensions - 2*x_toCrop
	y_toCrop = y_dimensions//10
	y_cropUntil = y_dimensions - 2*y_toCrop
	# bestScore = float("inf")
	bestScore = 0
	bestDisplacement = (0,0)
	secondPlate = secondPlate[x_toCrop:x_cropUntil, y_toCrop:y_cropUntil]


	for y in range(-15, 16):
		for x in range(-15, 16):
			tempPlate = firstPlate
			tempPlate = np.roll(tempPlate, y, axis = 1)
			tempPlate = np.roll(tempPlate, x, axis = 0)
			tempPlate = tempPlate[x_toCrop:x_cropUntil, y_toCrop:y_cropUntil]
			val = NCC(tempPlate, secondPlate)
			if val>bestScore:
				bestScore = val
				bestDisplacement=(x,y)

	return bestDisplacement

def findDisplacement2(firstPlate, secondPlate, estimate):
	start_x = 2 * estimate[0]
	start_y = 2 * estimate[1]

	firstPlate = np.roll(firstPlate, start_x, axis = 0)
	firstPlate = np.roll(firstPlate, start_y, axis = 1)
	temp = firstPlate.shape
	x_dimensions = temp[0]
	y_dimensions = temp[1]
	x_mid = x_dimensions / 2
	y_mid = y_dimensions / 2

	border = min(150, x_dimensions/6)
	border = min(border, y_dimensions/6)

	x_toCrop = x_mid - border
	y_toCrop = y_mid - border
	x_cropUntil = x_mid + border
	y_cropUntil = y_mid + border

	bestScore = 0
	bestDisplacement = (0,0)
	secondPlate = secondPlate[x_toCrop:x_cropUntil, y_toCrop:y_cropUntil]

	for y in range(-5, 6):
		for x in range(-5, 6):
			tempPlate = firstPlate
			tempPlate = np.roll(tempPlate, y, axis = 1)
			tempPlate = np.roll(tempPlate, x, axis = 0)
			tempPlate = tempPlate[x_toCrop:x_cropUntil, y_toCrop:y_cropUntil]
			val = NCC(tempPlate, secondPlate)
			if val>bestScore:
				bestScore = val
				bestDisplacement=(x,y)
	return (bestDisplacement[0] + start_x, bestDisplacement[1] + start_y)



def recursiveAlign(a, b):
	if a.shape[0] < 500 and a.shape[1] < 500:
		return findDisplacement(a,b)
	else:
		#resize here
		small_a = cv2.resize(a, (0,0), fx = 0.5, fy = 0.5)
		small_b = cv2.resize(b, (0,0), fx = 0.5, fy = 0.5)
		estimate = recursiveAlign(small_a, small_b)
		return findDisplacement2(a, b, estimate)

def align(a,b):
	val = recursiveAlign(a, b)
	a = np.roll(a, val[0], axis = 0)
	a = np.roll(a, val[1], axis = 1)
	return a




def colorize(name):
	# name of the input file
	imname = name
	res_name = imname[:-4]

	# read in the image
	# im = skio.imread(imname)
	im = cv2.imread("data/" + imname)
	im = im.astype(np.float32)



	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# convert to double (might want to do this later on to save memory)        
	# compute the height of each part (just 1/3 of total)
	height = np.floor(im.shape[0] / 3.0)

	# separate color channels
	b = im[:height]

	g = im[height: 2*height]

	r = im[2*height: 3*height]

	ag = align(g, b)
	ar = align(r, b)
	im_out = np.dstack([b, ag, ar])
	im_out = im_out.astype(np.uint8)



	# save the image

	fname = res_name + "_colorized" + ".jpg"
	# skio.imsave(fname, im_out)
	cv2.imwrite(fname, im_out, [int(cv2.IMWRITE_JPEG_QUALITY), 90]);



# toRun = {"bridge.tif", "cathedral.jpg", "emir.tif", "harvesters.tif", "lady.tif", "melons.tif", "monastery.jpg", "onion_church.tif", "selfie.tif", "settlers.jpg", "three_generations.tif", "tobolsk.jpg", "train.tif", "turkmen.tif", "village.tif"}

toRun = os.listdir("data")

for pic in toRun:
	colorize(pic)