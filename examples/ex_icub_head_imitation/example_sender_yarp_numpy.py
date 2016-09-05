import numpy
import yarp
 
# Initialise YARP
yarp.Network.init()
 
# Create the array with random data
img_array = numpy.random.uniform(0., 255., (240, 320)).astype(numpy.float32)
 
# Create the yarp image and wrap it around the array  
yarp_image = yarp.ImageFloat()
yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
 
# Create the yarp port, connect it to the running instance of yarpview and send the image
output_port = yarp.Port()
output_port.open("/python-image-port")
yarp.Network.connect("/python-image-port", "/view01")
output_port.write(yarp_image)
 
# Cleanup
output_port.close()
