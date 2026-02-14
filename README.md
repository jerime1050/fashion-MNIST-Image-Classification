# Tagalog-Jerime_LW1_Image_Classification

##google colab link:[[https://colab.research.google.com/drive/1eNoyVSz73CJTTtI23won4TolYwnUvQMJ](https://colab.research.google.com/drive/1UBckB08lLFyktR0alqDpVAjRkmN54uZv?usp=sharing)
](https://colab.research.google.com/drive/1UBckB08lLFyktR0alqDpVAjRkmN54uZv#scrollTo=N-ieJZN9C5ID)
#Question:
###1. What is the Fashion MNIST dataset?
The Fashion MNIST dataset is a collection of small grayscale images that show different types of clothing. It is commonly used to train and test machine learning models, especially in image classification. The dataset contains 70,000 images in total, where 60,000 are used for training and 10,000 are used for testing. Each image is 28 by 28 pixels and belongs to one of ten clothing categories such as shirts, shoes, bags, and dresses. It was created to replace the original MNIST handwritten digit dataset and provide a slightly more realistic and challenging task.

###2. Why do we normalize image pixel values before training?
Image pixel values normally range from 0 to 255, which can be large for a neural network to process efficiently. Normalizing these values by converting them into a range between 0 and 1 helps the model learn faster and perform better. It also makes training more stable because smaller numbers are easier for the algorithm to handle. This process helps the model adjust its weights more smoothly during learning.
###3. List the layers used in the neural network and their functions.
The neural network usually starts with a Flatten layer, which converts the 2D image into a single list of numbers so it can be processed by the network. After that, there is a Dense or hidden layer that helps the model learn patterns such as edges, shapes, and textures in the images. This layer often uses an activation function like ReLU to allow the model to learn complex relationships. Lastly, there is an output Dense layer that produces the final prediction. It usually has ten neurons, one for each clothing category, and uses the Softmax activation function to determine the probability of each category.
###4. What does an epoch mean in model training?
An epoch refers to one complete cycle where the model processes all the training data once. During each epoch, the model learns and adjusts its parameters based on the data it sees. Training for multiple epochs allows the model to improve its accuracy, although too many epochs may cause overfitting, where the model performs well on training data but poorly on new data.

###5. Compare the predicted label and actual label for the first test image.
The actual label represents the correct category of the image from the dataset, while the predicted label is the category guessed by the model. If both labels are the same, it means the model made a correct prediction. If they are different, it means the model misclassified the image. The comparison helps evaluate how well the model is performing.
###6. What could be done to improve the model’s accuracy?
The model’s accuracy can be improved in several ways. Increasing the number of training epochs can help the model learn more from the data. Adding more hidden layers or neurons can also improve the model’s ability to recognize patterns. Using Convolutional Neural Networks (CNNs) can significantly improve image classification performance. Other techniques include adjusting hyperparameters such as learning rate and batch size, applying dropout to prevent overfitting, and using data augmentation to provide more training variations.


