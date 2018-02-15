# Machine_Learning
Machine Learning Algorithms using Python

**Regression, Classification and KNN** <br />
**1. Linear Regression:**<br />
Developed a regularized Linear Regression model to predict Wine Quality. Trained UCI Wine Quality Dataset and used validation set to tune regularized hyper-parameter lambda giving lowest mean square error.<br />
**2. Logistic Regression:**<br />
Implemented gradient descent to train parameters of logistic regression model. Used MNIST dataset of handwritten digits from 0 to 9. The implemented model is only able to classify the dataset linearly, i.e., classifying digits less than 5 and digits more than 5.<br />
**3. K Nearest Neighbors:**<br />
Implemented kNN classifier to classify handwritten digits from MNIST dataset into 0 to 9. Found best k giving the best classification accuracy on the validation set. <br />

**Multinomial Logistic Regression, Neural Networks, Convolutional NN** <br />
**1. Multinomial Logistic Regression:**<br />
Developed a Multinomial Logistic Regression to classify images of handwritten digits into 10 classes (0 to 9). <br />
**2. Neural Networks:**<br />
Implemented a multi-layer perceptron having 1 hidden layer using rectified linear unit and softmax functions. Parameters are learnt using back propagation algorithm and classified images of handwritten digits into 10 classes (0 to 9).<br />
**3. CNN:**<br />
Implemented Convolutional NN classifier using ReLu, softmax and max-pooling functions to classify handwritten digits from MNIST dataset into 0 to 9. Learnt the parameters using back propagation algorithm.<br />

**Pegasos** <br />
Developed a stochastic gradient based solver for a linear Support Vector Machine. Built a SVM classifier model to classify images of handwritten digits into 10 classes (0 to 9).<br />

**Naive Bayes, K means Clustering, Expectation-Maximization** <br />
**1. Naive Bayes:**<br />
Built a binary spam-ham classifier using a naive Bayes model which can classify each document into spam class or ham class. Used smoothing technique to handle probability of unseen words. <br />
**2. K means Clustering:**<br />
Implemented K-means Clustering model to classify the dataset into clusters. K means algorithm tries to minimize the objective function. <br />
**3. EM to fit GMM:**<br />
Implemented an Expectation Maximization algorithm to fit a Gaussian Mixture Model and classify dataset into K clusters. Achieved accuracy of 98% with K=3. <br />

**HMM, PCA, t-SNE** <br />
**1. HMM:**<br />
Implemented Hidden Markov Model using transition probabilities from one state to another and observation probabilities of each observation given a state.<br />
**2. PCA:**<br />
Implemented a dimensionality reduction technique Principal Component Analysis.<br />
**3. t-SNE:**<br />
Implemented a non-linear dimensionality technique t-SNE. <br />
