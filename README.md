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
(128 * 128?) (June 28th)

2. Train ResNet-152 and do ensemble with DenseNets (Done) (June 23rd)

3. Use majority voting ensemble method with all ResNet models and DenseNet models (June 26th).

4. Use the IR channel. (June 27th)

--------------------------------------
## Update June 23

1. Trained ResNet152

2. Used the ensemble of ResNet152, DenseNet161, and DenseNet169 to do predictions on test set. (LB: 0.93068, Rank: 23)

3. Used different transformations on test dataset and average the ensemble results. (LB: 0.93112, Rank: it is 18th now :(  )

---------------------------------------

## Update June 28

Majority voting gives a small improvement: 0.93112 -> 0.93114. Now rank at 21st.

----------------------------------------
## Update July 2

Majority voting with fine-tuning the full dataset on pre-trained models gives me rank at 19th with f2 score 0.93170.

---------------------------

Have been doing some other stuff during the weekend, no progress yet. If anyone wants to teamup please email me at jxu7@bsu.edu. In addition, if anyone has new ideas using my code, please discuss in the Kaggle discussion board so that every competitor is able to learn some new stuff! Thanks!
