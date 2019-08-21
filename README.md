# ML4MI_BootCamp
Code from the UW-Madison Machine Learning for Medical Imaging (ML4MI) Boot Camp. For more information about ML4MI go to:  https://ml4mi.wisc.edu/

All the excercises are written in Keras which is integrated into tensorflow. We use the Keras functional model which is a lot more flexible than the commonly used sequential model in examples. 

Keras Documentations:
    https://keras.io/
    
Bootcamp organizers:
- Alan McMillan ( AMcmillan@uwhealth.org )
- Jacob Johnson ( jmjohnson33@wisc.edu ) 
- Kevin Johnson  ( kmjohnson3@wisc.edu )
- Tyler Bradshaw  ( tbradshaw@wisc.edu )

# System Requirements (Standalone)
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
These have been adapted to the Google research supported Colab. This is a free cloud based enviroment supported by Google. You can click on the link in the source code or go to https://colab.research.google.com/ and open from github. For UW-Madison users, you need to have access to the ML4MI_BOOTCAMP_DATA Google Drive folder.

# Examples:
- FunctionFitting - Some very basic networks used for learning functions 
- ImageReconstruction - Training of an neural network to reconstruct MRI images using 1D operations 
- MaleFemaleRadiograph - Classify chest xrays as male or female
- ImageSegmentation - Lung segmentation from CT data (need to download data yourself)
- AgeRegression - Regression for Age 
- ImageSynthesis - Image synthesis of brats data

# Note on commits:
If you aim to push changes to this repository, please edit the jupyter notebooks and then run clear_and_convert_bash (in linux or WSL). This will convert the notebooks to python and clear the ouput.

