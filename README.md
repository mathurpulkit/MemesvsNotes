<!DOCTYPE html>
<html>
<body>
<p>Memes vs Notes Classifier is a binary classifier used to identify whether a given image is a meme or notes.
</p>
<p>It has three different options available for training the neural network:</p>
<ol type="a">
<li>CNN based model using TensorFlow: It reads the images as 256*256*3 array. It uses three Convolution layers
 with 16,32,64 filters and 3,3,5 kernel size respectively. All Convolution layers use ReLU activation function.
  Each Convolution layer if followed by max pooling layer with stride of 2 and pool size of 2,2. 
  Then the output matrix is flattened into a 65536-D vector.
  This is connected to a network of fully connected layers with 500,200,80 and 2(output layer) respectively.
  All of the fully connected layers use ReLU except the output layer, which uses softmax. 
  All fully connected layers also use dropout regularization.
  Train Set Accuracy is 98.9%, Test Set accuracy is 98.4%.</li>
 <li>Fully Connected Using TensorFlow: It reads the images as 64*64*3 array.
 It uses 1000, 400, 100, 40, 2(output layer) and uses ReLU for every layer except the output layer, which uses softmax. 
 All fully connected layers also use dropout regularization.
 Train set accuracy is 93.2%, Test Set Accuracy is 95.3%. 
 </li>
 <li>Fully Connected using NumPy only: It reads the images as 64*64*3 array. It uses 1000, 400, 100, 40, 2(output layer) and uses ReLU for every layer except the output layer, which uses sigmoid.
 It uses L2 regularization.
 Train set accuracy is 95.8%, Test Set Accuracy is 90.6%. 
 </li>
</ol>
</body>
</html>