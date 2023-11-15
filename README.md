# LIST OF EXPERIMENTS:

|    | TITLE                                                                                            | PROGRAM                                      | DATASET                                        |
|----|--------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------|
| 1  | Solving XOR problem using Multilayer perceptron.                                                 | [Program](1-XOR-USING-MLP.ipynb)             |                                                |
| 2  | Implement character and Digit Recognition using ANN.                                             | [Program](2-ANN-CHARACTER-RECOGNITION.ipynb) |                                                |
| 3A | Implement the analysis of handwritten images using auto-encoders.                                | [Program](3A-HANDWRITTEN-AUTOENCODERS.ipynb) |                                                |
| 3B | Implement the analysis of Medical XRAY image classification using CNN.                           | [Program](3B-XRAY-CLASSIFICATION.ipynb)      | [Dataset](datasets/3B-XRAY-CLASSIFICATION.zip) |
| 4  | Implement Speech Recognition using NLP.                                                          | [Program](4-SPEECH-RECOGNITION-NLP.ipynb)    | [Dataset](datasets/4-SPEECH-RECOGNITION.zip)   |
| 5  | Develop a code to design object detection and classification for traffic analysis using CNN.     | [Program](5-TRAFFIC-ANALYSIS-CNN.ipynb)      | [Dataset](datasets/5-TRAFFIC-ANALYSIS-CNN.zip) |
| 6  | Implement online fraud detection of share market data using any one of the data analytics tools. | [Program](6-ONLINE-FRAUD-DETECTION.ipynb)    | [Dataset](datasets/6-FRAUD-DETECTION.zip)      |
| 7A | Implement image augmentation using TensorFlow                                                    | [Program](7A-IMAGE-AUGMENTATION.ipynb)       |                                                |
| 7B | Implement RBM Modeling to understand Hand Written Digits.                                        | [Program](7B-RBM-MODELING.ipynb)             |                                                |
| 8  | Implement Sentiment Analysis using LSTM.                                                         | [Program](8-SENTIMENT-ANALYSIS-LSTM.ipynb)   |                                                |

# NOTE:

    Please extract the datasets if they are in zip format.
    For example: 3B-XRAY-CLASSIFICATION.zip should be extracted as 3B-XRAY-CLASSIFICATION folder.

# Getting CUDA errors ?

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Add the above code in the first cell of the notebook to run the code on CPU. And also please restart the Jupyter Notebook Kernel after adding the above code.
