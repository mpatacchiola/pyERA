
Python implementation of the Epigenetic Robotic Architecture (ERA). It includes standalone classes for Self-Organizing Maps (SOM) and Hebbian Leaning.


What is ERA?
------------

The Epigenetic Robotic Architecture (ERA) is a hybrid behavior-based robotics and neural architecture purposely built to implement embodied principles in cognitive development. This architecture has been already tested in a variety of cognitive and developmental tasks directly modeling child psychology data. The ERA architecture uses a behaviour-based subsumption mechanism to handle the integration of competing sensorimotor input. The learning system is based on an ensemble of pre-trained Self-Organizing Maps (SOMs) connected via Hebbian weights. Various SOMs perform vision, motor and speech classification tasks and learn hierarchical associations between modalities. During interaction with the users in a learning phase, the connections weights between different modalities are trained via Hebbian rule. 

Installation
------------

The package does not require any special library, the only requirement is numpy. In the following instruction I suppose that you cloned pyERA in your home folder. To import the pyERA modules it is possible to set the package directory in the $PYTHONPATH from terminal: 

```shell
export PYTHONPATH="${PYTHONPATH}:~/pyERA/pyERA"
```

When you restart the terminal you have to export again the package. Alternatively you can write the same line in your `~/.bashrc` file. You can also load the package directly from python code, inserting the following lines at the beginning of the file:

```python
import sys
sys.path.insert(0, "~/pyERA/pyERA")
```

If the package was correctly added to your $PYTHONPATH you can use it and import the different pyERA modules, for example:

```python
from pyERA.som import Som
from pyERA.utils import ExponentialDecay
```



