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

## Installation  
1. Clone this repository:  
   `git clone https://github.com/Richielee630/Fashion-MNIST-RLI.git`
3. Install the necessary dependencies:  
   `pip install -r requirements.txt`

## Usage  
1. **Data Preparation**: Download the Fashion MNIST dataset from Kaggle and place it in the `data/` directory.  
2. **Training the Model**: Run the following command to start training:  
   `python train_model.py`  
3. **Evaluation**: After training, evaluate the model's performance on the test dataset:  
   `python evaluate_model.py`  
4. **Inference**: To classify a new image, use the inference script:  
   `python infer.py --image_path path/to/image.png`

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

