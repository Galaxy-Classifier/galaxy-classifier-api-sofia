from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt


class Classifier():  
  def makePrediction(self,image_file,cnn_model, autoencoder_models):
    #Declaracion de variables
    #Size of width & height 
    size =100
    #Etiquetas de las clases 
    label_classes = ['elliptical','espiral','lenticular']
    #variables para regresar el ganador
    winner_label = ""
    winner_value = 0
    #Recibe una image y la convierte en arreglo de np
    data = []
    npimg = np.fromstring(image_file, dtype=np.uint8); 
    image = cv2.imdecode(npimg,1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_resized = cv2.resize(gray_image, (size, size), interpolation=cv2.INTER_LINEAR)
    data.append(im_resized)
    image_array = np.array(data)
    image_array = np.concatenate([image_array])
    img = image_array.reshape(1, size, size, 1).astype('float32')
    img /= 255


    #Definimos los labels 
    lbl= LabelBinarizer()
    labels = lbl.fit_transform(label_classes)
    #Recorremos todos los autoencoders y realizamos predicciones
    for model in autoencoder_models:
      #Pasamos la imagen por el autoencoder
      autoencoder_result= model.predict(img)
      #Pasamos la imagen obtenida por el autoencoder por el modelo de la cnn
      cnn_result = cnn_model.predict(autoencoder_result)
      #Obtener etiqueta de la prediccion
      label_result = lbl.inverse_transform(cnn_result)
      #Obtener el valor del resultado mas cercano a 1 
      array = np.asarray(cnn_result[0])
      idx = (np.abs(array - 1.0 )).argmin()
      value = array[idx]
      #Comparar con el valor ganador para guardarlo como la mejor prediccion
      if value > winner_value :
        winner_label = label_result[0]
        winner_value = value
      #Prints axuliares
    #   print(model_path)
    #   print("Resultados de la cnn")
    #   print(cnn_result)
    #   print("Label ganador este autoencoder")
    #   print(label_result[0])
    #   print("Valor del ganador de este autoencoder")
    #   print(value)
    
    # print(winner_label)
    # print(winner_value)
    return winner_label