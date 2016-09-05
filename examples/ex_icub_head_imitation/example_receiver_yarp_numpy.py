import numpy
import yarp
#import matplotlib.pylab
import matplotlib.pyplot as plt
import time 
import cv2

# Initialise YARP
yarp.Network.init()
 
# Create a port and connect it to the iCub simulator virtual camera
input_port = yarp.Port()
input_port.open("/python-image-port")
yarp.Network.connect("/icubSim/cam/left", "/python-image-port")
 
# Create numpy array to receive the image and the YARP image wrapped around it
img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
yarp_image = yarp.ImageRgb()
yarp_image.resize(320, 240)
yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
 
# Read the data from the port into the image
input_port.read(yarp_image)
 
# display the image that has been read
#matplotlib.pylab.imshow(img_array)
print("Showing the image...")
#imgplot = plt.imshow(img_array)
cv2.imshow('image',img_array)
cv2.waitKey(0)
cv2.destroyAllWindows()

#time.sleep(3)
# Cleanup
input_port.close()
