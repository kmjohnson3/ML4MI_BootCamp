# ML4MI_BootCamp
Code from the UW-Madison Machine Learning for Medical Imaging (ML4MI) Boot Camp. For more information about ML4MI go to: https://www.radiology.wisc.edu/research/medical-imaging-machine-learning-initiative/ 

All the excercises are written in Keras which is corrently integrated into tensorflow. We use the Keras functional model which is a lot more flexible than the commonly used sequential model in examples. 

Keras Documentations:
    https://keras.io/
    
Bootcamp organizers:
- Alan McMillan ( AMcmillan@uwhealth.org )
- Jacob Johnson ( jmjohnson33@wisc.edu ) 
- Kevin Johnson  ( kmjohnson3@wisc.edu )
- Tyler Bradshaw  ( tbradshaw@wisc.edu )

# System Requirments
Code has been tested on a machine with a NVIDIA K80 (11gb of GPU ram). To run this you need:
python 3 ( https://www.python.org/ )
tensorflow ( https://www.tensorflow.org/install/ , install tensorflow-gpu if you have one)

We installed these with the following commands.
```bash
pip install tensorflow-gpu
pip install keras
pip install matplotlib
pip install numpy
pip install livelossplot
pip install conda
pip install jupyterlab
conda install scikit-image
conda install scipy
conda install -c conda-forge --no-deps pydicom
```

# Colab from Google Research
Some of these will run on the Google research supported Colab. This is a free cloud based enviroment supported by Google. You can click on the link in the source code or go to https://colab.research.google.com/ 

# Working Examples:
- FunctionFitting - Some very basic networks used for learning functions 
- ImageReconstruction - Training of an neural network to reconstruct MRI images using 1D operations 

# Examples missing data (work in progress):
- MaleFemaleRadiograph - Classify chest xrays as male or female
- AgeRegression - Regression for Age 
- ImageSegmentation - Lung segmentation from CT data 
- ImageSynthesis - Image synthesis of brats data

