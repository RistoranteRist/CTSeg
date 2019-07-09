# CTSeg

This repository contains the Python code for automatic segmentation of internal organs from male pelvic CT images.

## Learning


## Prediction

Using the model's weight produced as described in the previous paragraph, one can predict the contours of internal organs from CT images by using the class `CTSegPredictor` in the Python script `Prediction/CTSegPredictor.py`.

After adding a Dicom file named `CT.dcm` in the `Predictor` folder, using the command `python CTSegPredictor.py` in this folder will execute the prediction on this file with the sample model `Prediction/CTSegModel.hdf5` and output the result image `contours.png` in the same folder. The provided h5 model has been trained on axial prone CT images of the pelvis area to predict the bladder, prostate, seminal vesicle and colon organs.

For one to implement his/her own prediction code we suggest to use the method `ct_to_contours_png()` as a template.

Required modules are `os, sys, cv2, numpy, datetime, pydicom, PIL, tensorflow, pathlib, keras`. The code has been successfully tested with `Python 3.6.8, CV2 3.4.4, Numpy 1.16.3, Pydicom 1.2.2, PIL 5.1.0, Tensorflow 1.12.0, Keras 2.2.4` on a PC running Ubuntu 18.04.
