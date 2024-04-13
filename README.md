# Language Identification App

This repository contains a language identification app that can determine the language of a given input.

## Introduction

This project utilizes the [WiLI-2018 dataset](https://paperswithcode.com/dataset/wili-2018), which serves as a benchmark for monolingual written natural language identification. WiLI-2018 is a publicly available dataset comprising short text extracts from Wikipedia, encompassing 1000 paragraphs across 235 languages, totaling 23500 paragraphs. For simplicity and resource efficiency, we've narrowed down the selection to 22 languages, including English, Arabic, French, Hindi, Urdu, Portuguese, Persian, Pushto, Spanish, Korean, Tamil, Turkish, Estonian, Russian, Romanian, Chinese, Swedish, Latin, Indonesian, Dutch, Japanese, and Thai. 

WiLI is structured as a classification dataset, tasked with identifying the dominant language within an unknown paragraph. Our approach involves employing various deep learning techniques for language identification. For preprocessing, we utilize CountVectorizer to convert text into numerical features, LabelEncoder to encode language labels, and apply text cleaning techniques, including removing stopwords, punctuation, and stemming. Finally, it returns the processed data along with the encoded labels.

## Method

Initially, we employed a base neural network (Naive Bayes) model architecture for classification tasks. It comprises fully connected layers with ReLU activation and softmax output. The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.

| Layer          | Output Size | Activation | Initialization | Input Shape |
|----------------|-------------|------------|----------------|-------------|
| Input          | 100         | ReLU       | He_normal      | (input_size,) |
| Hidden Layer 1 | 80          | ReLU       | He_normal      | (100,)       |
| Hidden Layer 2 | 50          | ReLU       | He_normal      | (80,)        |
| Output         | output_size | Softmax    | N/A            | (50,)        |

Subsequently, we modified our method by employing a neural network model with three hidden layers, each containing 100, 80, and 50 neurons, respectively. We utilized the softsign activation function and glorot uniform kernel initialization. The output layer employs softmax activation. The model is compiled using the Adam optimizer and sparse categorical crossentropy loss.

| Layer          | Output Size | Activation | Initialization | Input Shape |
|----------------|-------------|------------|----------------|-------------|
| Input          | 100         | Softsign   | Glorot uniform | (input_size,) |
| Hidden Layer 1 | 80          | Softsign   | Glorot uniform | (100,)       |
| Hidden Layer 2 | 50          | Softsign   | Glorot uniform | (80,)        |
| Output         | output_size | Softmax    | N/A            | (50,)        |

## Results

After 5 epochs, the modified model achieved 99% accuracy on the training data and nearly 97% accuracy on the validation data. You can visualize the accuracy improvement and decreasing categorical crossentropy process in the following figures:

![Accuracy Plot](https://github.com/Anndischeh/Language_Identifier/blob/692b87172ee07c3e2e2720e41ad23f6e15abbf2b/media/ACC.png)  
![Loss Plot](https://github.com/Anndischeh/Language_Identifier/blob/692b87172ee07c3e2e2720e41ad23f6e15abbf2b/media/CC.png)

Additionally, you can view the confusion matrix for the modified model:

![Confusion Matrix](https://github.com/Anndischeh/Language_Identifier/blob/692b87172ee07c3e2e2720e41ad23f6e15abbf2b/media/NLP-CM.png)

## Example

Now, let's see an example of the model's prediction:

```python
predict('Hello world')
>> English

predict('سلام دنیا')
>> Persian
```

Feel free to explore further examples and functionalities provided by the language identification app!
