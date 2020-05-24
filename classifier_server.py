from concurrent import futures
import logging
import grpc
import classifier_pb2
import classifier_pb2_grpc
import classifier
import io
import base64
from PIL import Image



class Clasification(classifier_pb2_grpc.ClassifierServicer):

    def GetClassification(self, request, context):
        results = list()
        for obj in request.classificationRequest:
            image_decoded= base64.b64decode(obj.chunk_data)
            result = classifier.Classifier().makePrediction(image_decoded)
            results.append({"id": obj.id,"result": result})
        
        return classifier_pb2.ClassificationReply(classificationResponse=results)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    classifier_pb2_grpc.add_ClassifierServicer_to_server(Clasification(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
