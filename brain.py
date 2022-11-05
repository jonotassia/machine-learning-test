from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd() + "\\assets"


prediction = ImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(
    execution_path, "resnet50_imagenet_tf.2.0.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "godzilla.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
