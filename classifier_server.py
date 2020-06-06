from concurrent import futures
import logging
import grpc
import classifier_pb2
import classifier_pb2_grpc
import classifier
import io
import base64
import glob
from PIL import Image
from tensorflow.keras.models import load_model
autonencoders_path_folder = "./autoencoders/*"
cnn_path_model = "./cnn/galaxiasCNN.h5"
cnn_model = load_model(cnn_path_model)
autonencoders_models = list()


class Clasification(classifier_pb2_grpc.ClassifierServicer):

    def GetClassification(self, request, context):
        if cnn_model == None or len(autonencoders_models) == 0 :
            loadModels()  
        results = list()
        for obj in request.classificationRequest:
            image_decoded= base64.b64decode(obj.chunk_data)
            result = classifier.Classifier().makePrediction(image_decoded,cnn_model,autonencoders_models)
            results.append({"id": obj.id,"result": result})
        
        return classifier_pb2.ClassificationReply(classificationResponse=results)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    classifier_pb2_grpc.add_ClassifierServicer_to_server(Clasification(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def loadModels():
    paths = glob.glob(autonencoders_path_folder)
    for path in paths :
        model = load_model(path)
        autonencoders_models.append(model)
    print('Modelos cargados')
if __name__ == '__main__':
    logging.basicConfig()
    loadModels()
    serve()
    

    
