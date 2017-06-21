## Some thoughts so far:

1. Deep neural nets like DenseNet or ResNet are good for computer 
vision even for moderate dataset (like this competition: ~45000)

2. While stochastic gradient descent with momentum needs more time to 
converge, with a good learning rate setup, it is able to outperform Adam. (why??)

3. Fine-tuning pre-trained deep model usually works better than training from scratch. (why??)

4. Ensemble method is a good way to increase accuracy 
(maybe different models learn different features, by averaging these differences, it reduces biases and variances.)


## TODO:

1. Finish feature pyramid network with group convolution 
(Xception-like network?) and train it on a smaller sized image. 
(128 * 128?)

2. Train ResNet-152 and do ensemble with DenseNets

3. Use the IR channel.