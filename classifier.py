from tensorflow.keras.models import load_model
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
size =100

url_model_elliptical_autoencoder = "./models/autoencoder_elliptical.h5"
url_model_espiral_autonencoder = "./models/autoencoder_espiral.h5"
url_model_lenticular_autonencoder = "./models/autoencoder_lenticular.h5"
url_model_cnn = "./models/galaxiasCNN.h5"

class Classifier():
    def makeClassification(self, imageFile):
        #Recibe una image y la convierte en arreglo de np
        data = []
        npimg = np.fromstring(imageFile, dtype=np.uint8); 
        image = cv2.imdecode(npimg,1)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        im_resized = cv2.resize(gray_image, (size, size), interpolation=cv2.INTER_LINEAR)
        data.append(im_resized)
        image_array = np.array(data)
        image_array = np.concatenate([image_array])

        #Cargamos los tres modelos de autoencoder
        elliptical_autoenconder = load_model(url_model_elliptical_autoencoder)
        espiral_autoenconder = load_model(url_model_espiral_autonencoder)
        lenticular_autoenconder = load_model(url_model_lenticular_autonencoder)
        img = image_array.reshape(1, size, size, 1).astype('float32')
        img /= 255

        #Pasamos la imagen por los tres autoencoders
        encode_img_elliptical = elliptical_autoenconder.predict(img)
        encode_img_espiral = espiral_autoenconder.predict(img)
        encode_img_lenticular = lenticular_autoenconder.predict(img)

        #Predecimos la red neuronal con los tres images obtenidas de los autoencoders
        cnn = load_model(url_model_cnn)
        elliptical_pred = cnn.predict(encode_img_elliptical)
        espiral_pred = cnn.predict(encode_img_espiral)
        lenticular_pred = cnn.predict(encode_img_lenticular)

        return elliptical_pred