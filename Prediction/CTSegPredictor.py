# coding: utf-8
# ============= CTSegPredictor.py ===============

# Import third party modules
import os
import sys
import cv2
import numpy
import datetime
import pydicom as dicom
from PIL import Image
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers.pooling import MaxPool2D
from keras.layers import Input
from keras.layers.merge import add
from keras.layers.convolutional import Conv2D
from keras.layers import BatchNormalization
from keras.layers import ReLU
from keras.layers.convolutional import Conv2DTranspose


# Root folder of the script
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# To avoid the error "Tensor is not an element of this graph" when
# using the method CTSegPredictor.prediction(), memorize the default
# graph of TensorFlow
global tf_default_graph
tf_default_graph = tf.get_default_graph()


def log_err(exc):
    """
    Helper function to display error messages
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print("CTSegPredictor: " +
          str(datetime.datetime.now()) + " " +
          str(exc_type) + " " +
          str(fname) + " " +
          str(exc_tb.tb_lineno) + " " +
          str(exc))


class CTSegPredictor:
    """
    Class implementing the prediction of contours of organs in a CT
    image
    """

    def __init__(self, nb_organs, path_model=None, threshold=0.5):
        """
        Constructor
        Input:
            'nb_organs': int, number of organs predicted by the model
            'path_model': pathlib.Path, path to the model's weights
            'threshold': float in [0.,1.], threshold to discriminate the
                pixels in the result of prediction. If the pixel value
                is above the threshold it belongs to the corresponding
                organ.
        """
        try:

            # Path to the model
            self.path_model = None

            # Nb of convolution in the model
            self.nb_channels = 32

            # Nb of organs predicted by the model
            self.nb_organs = nb_organs

            # Input image dimension in the CT input (the image is
            # square)
            self.img_size = 512

            # Threshold to discriminate the
            # pixels in the result of prediction
            self.threshold = threshold

            # Model instance
            self.model = self.get_fusionnet()

            # If the user provided a model, load it
            if path_model:
                self.load_model(path_model)

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.__init__")

    def load_model(self, path_model):
        """
        Load the trained model's weight
        Input:
            'path_model': pathlib.Path, path to the h5 model file
        """
        try:

            print("CTSegPredictor: Load the model " + str(path_model))

            # Check that the path is correct
            if not path_model.exists():
                raise UserWarning("The path is invalid")

            # Memorize the path to the used model
            self.path_model = path_model

            # Load the weights
            self.model.load_weights(str(self.path_model))

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.load_model")

    def get_fusionnet(self):
        """
        Create the FusionNet model
        """
        try:

            print("CTSegPredictor: Create the fusionNet model")

            # Get the input dimensions
            dim_input = (self.img_size, self.img_size, 1)

            # Create the encoding layers
            inputs = Input(shape=dim_input)
            y1 = self.residual(inputs,
                               self.nb_channels)
            y2 = MaxPool2D(pool_size=(2, 2),
                           padding="same")(y1)
            y2 = self.residual(x=y2,
                               nb_channels=(self.nb_channels * 2))
            y3 = MaxPool2D(pool_size=(2, 2),
                           padding="same")(y2)
            y3 = self.residual(x=y3,
                               nb_channels=(self.nb_channels * 4))
            y4 = MaxPool2D(pool_size=(2, 2),
                           padding="same")(y3)
            y4 = self.residual(x=y4,
                               nb_channels=(self.nb_channels * 8))
            y5 = MaxPool2D(pool_size=(2, 2),
                           padding="same")(y4)
            y5 = self.residual(x=y5,
                               nb_channels=(self.nb_channels * 16))

            # Create the decoding layers
            y6 = Conv2DTranspose(filters=(self.nb_channels * 8),
                                 kernel_size=3,
                                 strides=2,
                                 kernel_initializer="he_normal",
                                 padding="same",
                                 activation="relu")(y5)
            y6 = add([y6, y4])
            y6 = self.residual(x=y6,
                               nb_channels=(self.nb_channels * 8))
            y7 = Conv2DTranspose(filters=(self.nb_channels * 4),
                                 kernel_size=3,
                                 strides=2,
                                 kernel_initializer="he_normal",
                                 padding="same",
                                 activation="relu")(y6)
            y7 = add([y7, y3])
            y7 = self.residual(x=y7,
                               nb_channels=(self.nb_channels * 4))
            y8 = Conv2DTranspose(filters=(self.nb_channels * 2),
                                 kernel_size=3,
                                 strides=2,
                                 kernel_initializer="he_normal",
                                 padding="same",
                                 activation="relu")(y7)
            y8 = add([y8, y2])
            y8 = self.residual(x=y8,
                               nb_channels=(self.nb_channels * 2))
            y9 = Conv2DTranspose(filters=self.nb_channels,
                                 kernel_size=3,
                                 strides=2,
                                 kernel_initializer="he_normal",
                                 padding="same",
                                 activation="relu")(y8)
            y9 = add([y9, y1])
            y9 = self.residual(x=y9,
                               nb_channels=self.nb_channels)
            y10 = Conv2D(filters=self.nb_organs,
                         kernel_size=1,
                         kernel_initializer="he_normal",
                         padding="same",
                         activation="sigmoid")(y9)

            # Create and return the model
            model = Model(inputs=inputs,
                          outputs=[y10],
                          name='fusion_net')
            return model

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.get_fusionnet")

    def residual(self, x, nb_channels):
        """
        Helper function for get_fusionnet
        """
        try:

            x = Conv2D(filters=nb_channels,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer="he_normal")((x))
            x = BatchNormalization()(x)
            x = ReLU()(x)
            y = Conv2D(filters=nb_channels,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer="he_normal")((x))
            y = BatchNormalization()(y)
            y = ReLU()(y)
            y = Conv2D(filters=nb_channels,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer="he_normal")((y))
            y = BatchNormalization()(y)
            y = ReLU()(y)
            x = add([x, y])
            x = Conv2D(filters=nb_channels,
                       kernel_size=3,
                       padding="same",
                       kernel_initializer="he_normal")((x))
            x = BatchNormalization()(x)
            x = ReLU()(x)
            return x

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.residual")

    def prediction(self, slices):
        """
        Do the prediction on a set of slices
        Input:
            'slices': numpy array of shape (n, w, h, c) and type
                float32 where 'n' is the number of slices, 'w, h' are
                the dimensions of the slices, and 'c' is the number
                of channel (should be 1, ie greyscale).
                Values are equal to (dicom.pixel_array / 255.0 ** 2)
        Output:
            Return an array of shape (n, w, h, o) where 'n' is the
            number of slices, 'w, h' are the dimensions of the
            slices, and 'o' is the number of organs predicted. The
            values for each organ are in [0., 1.], 0. means the pixel
            doesn't match the organ, 1. means the pixel matches the
            organ
        """
        try:
            print("CTSegPredictor: Prediction of " +
                  str(slices.shape[0]) + " slice(s)")

            # Run the prediction and return the result
            res_pred = None
            with tf_default_graph.as_default():
                res_pred = self.model.predict(slices,
                                              verbose=False)
            return res_pred

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.prediction")

    def get_pred_slice_organ(self, res_pred, i_slice, i_organ):
        """
        Get the data about a given slice and organ in the result of
        prediction
        Input:
            'res_pred': numpy array, result of CTSegPredictor.prediction
            'i_slice': int, index of the requested slice
            'i_organ': int, index of the requested organ
        Output:
            Return an array of shape (w, h) where 'w, h' are the
            dimensions of the slices. The values are in [0., 1.],
            0. means the pixel doesn't match the organ,1. means the
            pixel matches the organ
        """
        try:

            return res_pred[i_slice][:, :, i_organ]

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.get_pred_slice_organ")

    def is_organ_in_slice(self, res_pred, i_slice, i_organ):
        """
        Return true if a given organ has been detected in a given slice
        Input:
            'res_pred': numpy array, result of CTSegPredictor.prediction
            'i_slice': int, index of the requested slice
            'i_organ': int, index of the requested organ
        Output:
            Return True if there is at least one pixel in the result
            of prediction for the given slice and organ whose value is
            above self.threshold
        """
        try:
            pred = self.get_pred_slice_organ(res_pred,
                                             i_slice,
                                             i_organ)
            return numpy.any(pred > self.threshold)

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.is_organ_in_slice")

    def ct_to_contours_png(self, path_ct, path_result):
        """
        Do the prediction on the slice in a CT file and save the
        result as a PNG image
        Input:
            'path_ct': pathlib.Path, path to the CT file
            'path_result': pathlib.Path, path to result PNG image
        Output:
            Save the result image which is the horizontal
            concatenation of the input slice and the visualisation in
            black and white of the result of prediction for each organ.
            White pixels represent the organ in the result.
        """
        try:

            print("CTSegPredictor: Prediction of " + str(path_ct))

            # Load the dicom data of the slice
            dicom_data = dicom.dcmread(fp=str(path_ct),
                                       force=True)

            # Get the pixel data of the input image
            px_arr = dicom_data.pixel_array

            # Convert the dicom data to the model's input format
            px_arr = numpy.array(px_arr).astype("float32") / 255. ** 2
            nb_slice = 1
            nb_channel = 1
            px_arr = px_arr.reshape(nb_slice,
                                    self.img_size,
                                    self.img_size,
                                    nb_channel)

            # Run the prediction and get the result for the first and
            # only slice
            res_pred = self.prediction(px_arr)

            # Get the resized and normalized version of the original
            # image as a PIL image
            img_orig = cv2.resize(dicom_data.pixel_array,
                                  (self.img_size, self.img_size))
            img_orig = img_orig * (255.0 / numpy.amax(img_orig))
            img_orig_pil = Image.fromarray(img_orig).convert("RGB")

            # Create the final image large enough to contain the
            # original image and the prediction results
            dim_img_final = \
                (self.img_size * (nb_organs + 1), self.img_size)
            img_final_pil = Image.new('RGB',
                                      dim_img_final)

            # Add the original image to the final image
            img_final_pil.paste(img_orig_pil, (0, 0))

            # For each organ
            for i_organ in range(self.nb_organs):

                # If the organ has been detected
                is_organ_in_slice = \
                    self.is_organ_in_slice(res_pred,
                                           i_slice=0,
                                           i_organ=i_organ)
                if is_organ_in_slice:

                    # Get the result of prediction for this organ
                    img_organ = \
                        self.get_pred_slice_organ(res_pred,
                                                  i_slice=0,
                                                  i_organ=i_organ)

                    # Create the image of the prediction for this
                    # organ. Normalize it for better visualization
                    img_organ = \
                        img_organ * 255.0 / numpy.amax(img_organ)
                    img_organ_pil = \
                        Image.fromarray(img_organ).convert("RGB")

                    # Add the image for this organ to the final image.
                    # Concatenate the original image and prediction
                    # results horizontally.
                    pos_organ_in_final_img = \
                        ((i_organ + 1) * self.img_size, 0)
                    img_final_pil.paste(img_organ_pil,
                                        pos_organ_in_final_img)

            # Save the final image
            img_final_pil.save(str(path_result))

            print("CTSegPredictor: Save result to " + str(path_result))

        except Exception as exc:
            log_err(exc)
            raise RuntimeError("CTSegPredictor.ct_to_contours_png")


if __name__ == '__main__':
    try:

        # Path to the model's weight
        path_model = BASE_DIR / "CTSegModel.hdf5"

        # Number of organs predicted by the model
        nb_organs = 4

        # Create the predictor
        predictor = CTSegPredictor(nb_organs, path_model)

        # Path to the CT file on which to run the prediction
        path_ct = BASE_DIR / "CT.dcm"

        # Path to the PNG image where to save the result of prediction
        path_result = BASE_DIR / "contours.png"

        # Run the prediction on the CT file and save the predicted
        # contours as a PNG image
        predictor.ct_to_contours_png(path_ct, path_result)

    except Exception as exc:
        log_err(exc)

# ============= end of CTSegPredictor.py ===============
