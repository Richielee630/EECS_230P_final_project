# Fashion MNIST Classification Project 
## [UCI MEng EECS 230] 
This project aims to develop a machine learning model to classify images from the Fashion MNIST dataset, which includes various clothing items like t-shirts, trousers, bags, and shoes. The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).

## Project Overview  
The primary goal of this project is to build a robust classification model that can accurately categorize fashion images into ten predefined classes. We’ll use deep learning techniques, leveraging popular frameworks like TensorFlow or PyTorch, to achieve high accuracy in identifying different clothing items.

## Dataset  
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with 60,000 images in the training set and 10,000 in the test set. Each image is 28x28 pixels, representing a single clothing item. The classes are as follows:  
- 0: T-shirt/top  
- 1: Trouser  
- 2: Pullover  
- 3: Dress  
- 4: Coat  
- 5: Sandal  
- 6: Shirt  
- 7: Sneaker  
- 8: Bag  
- 9: Ankle boot  

## Requirements
To run this project, you need the following Python libraries:

- `torch`: For building and training the neural network.
- `torchvision`: For dataset transformations and utilities.
- `matplotlib`: For visualizing training/validation loss and accuracy.
- `seaborn`: For plotting the confusion matrix.
- `scikit-learn`: For generating the classification report and confusion matrix.
- `numpy`: For numerical computations.

Install all dependencies using the command:
```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy

## Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Richielee630/Fashion-MNIST-RLI.git
   ```
<!-- 2. Install the necessary dependencies:  
   ```bash
   pip install -r requirements.txt
   ``` -->

### Download and Place the Dataset  
1. Download the Fashion MNIST dataset from Kaggle:  
   [https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)  

2. After downloading, extract the dataset files. Ensure you have the following files:  
   - `fashion-mnist_train.csv`  
   - `fashion-mnist_test.csv`  

3. Place the extracted files in the `dataset/` directory inside your project folder:  
   ```
   class_project/
   ├── dataset/
   │   ├── fashion-mnist_train.csv
   │   ├── fashion-mnist_test.csv
   ├── notebook/
   │   ├── fashion_mnist_cnn_training.ipynb
   ├── README.md
   ├── .gitignore
   ```

## Usage  
1. **Data Preparation**: Ensure the Fashion MNIST dataset is placed in the `dataset/` directory as described above.  

2. **Training and Evaluating the Model**: Open the Jupyter Notebook file located in the `notebook/` directory:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Navigate to `notebook/fashion_mnist__simple_cnn_training.ipynb` in the Jupyter interface.
   - Follow the steps in the notebook to train, evaluate, and visualize the model's performance.

3. **Inference**: Modify the last cell in the notebook to classify new images by loading the trained model and providing the path to your image.

## Model Architecture  
This project will experiment with various model architectures, including Convolutional Neural Networks (CNNs), to improve classification accuracy.

## Results  
We will track and document accuracy, loss, and other relevant metrics to evaluate the model's performance.

## Future Work  
- Hyperparameter tuning  
- Experimenting with different architectures  
- Adding data augmentation techniques

## Acknowledgments  
Special thanks to Zalando Research for providing the Fashion MNIST dataset.