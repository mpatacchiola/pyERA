import numpy
import matplotlib.pyplot as plt
import cv2
import yarp

'''
img_array = numpy.zeros((240, 320, 3), dtype=numpy.uint8)
#imgplot = plt.imshow(img_array)

cv2.imshow('image',img_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#yarp.Network.init() # Initialise YARP
#output_port = yarp.Port()
#output_port.open("/icubSim/head/rpc:i")
#position = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2 )
#output_port.write(position)

yarp.Network.init() # Initialise YARP

_rpc_client = yarp.RpcClient()
#_rpc_client.open("/icubSim/head/rpc:i")
_rpc_client.addOutput("/icubSim/head/rpc:i")
        

bottle = yarp.Bottle()
result = yarp.Bottle()

bottle.clear()
bottle.addString("set")
bottle.addString("pos")
bottle.addInt(1) #Specifies the head Joint
bottle.addInt(15) # Specifies the Joint angle

print ("Sending", bottle.toString())
_rpc_client.write(bottle, result)
print ("Return", result.toString())

bottle.clear()
bottle.addString("set")
bottle.addString("pos")
bottle.addInt(2) #Specifies the head Joint
bottle.addInt(0) # Specifies the Joint angle

print ("Sending", bottle.toString())
_rpc_client.write(bottle, result)
print ("Return", result.toString())

#result = yarp.Bottle()
#result.clear()
#result.addDouble, location


