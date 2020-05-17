from __future__ import print_function
import logging
import grpc
import classifier_pb2
import classifier_pb2_grpc
import os
import sys
import io
import base64
from PIL import Image
from array import array

def readimage(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read())

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = classifier_pb2_grpc.ClassifierStub(channel)
        img_bytes = readimage("/Users/victormorfin/Desktop/tecCuliacan/Residencias/galaxy-classifier-nn/data/Elliptical/ESO486-21.jpg") 
        response = stub.GetClassification(classifier_pb2.ClassificationRequest(id="1234"  ,image=img_bytes))
    print("Classifier client received: " + response.message)


if __name__ == '__main__':
    logging.basicConfig()
    run()