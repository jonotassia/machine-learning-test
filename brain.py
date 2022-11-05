from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd() + "/assets/img"


prediction = ImagePrediction()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(
    execution_path, "resnet50_imagenet_tf.2.0.hf"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(
    os.path.join(execution_path, "giraffe.jpg"), result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
