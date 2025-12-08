# ChallengeNN
 


This challenge runs a bit differently from your homework assignments. You may use the test code provided here to help you understand how we will run your code, but we will be testing all models **offline**.


Your challenge, should you choose to accept it, is to design neural networks to solve these problems. Your points will be calculated as follows.

 - Binary classification: points(model) = -log10(misclassification), where misclassification is between 0 and 1
 - K-class classification: points(model) = -log10(misclassification)
 - Regression: points(model) = max(0,-log10) (MSE) 
 
Your challenge consists of 2 parts. 

Part 1: We evaluate your *fully trained* model. 
Part 2: We take a model you have given, + training code, and train it on a *slightly different dataset*. It will have similar properties, but the class labels will either be scrambled, or the features will be altered in some other way, so you cannot hope for a good score using your trained model. We will run your train code, with a time-out of 1 minute, and evaluate this model on our test set. 

 
The final scoring will be: 
 
points per challenge: 1/2 (  points(trained model over your train dataset) + points(trained model over your test dataset) + points(model + train code over my train set) + points(model + train code over my test set) ).
 
Save your models as follows, inside a directory called 'models', pickled:  

challengetype_challengenumber_trainornot.pkl

where challenge_type is **binclass**, **multiclass**, or **regress**. For mnist and mapregress, just use **mnist** or **mapregress** and drop the challenge number when naming files. trainornot is **trained** or **untrained**.

Each pickled file should be saved using syntax like:
pickle.dump(model,open('binclass_1_trained.pkl','wb'))

Make sure that your trained models have a command for inference, e.g. yhat = model.inference(X). See the test file to see how the models will be run.

Save your train code so that I can call the commands

    def master_train_model(probnum):
        model = pickle.load(open('challenge%s_untrained.pkl' % probnum,'rb'))
        model = train(model, probnum, Xtrain, ytrain) # you write the train function
 
I should be able to access the `train` function after `from neuralNet.py import *`.

Extra data files:

https://www.dropbox.com/scl/fi/x9resobi2xne514jvnk1o/mapregress.pkl?rlkey=6rjra8nz4yl6moyevsrdc2lzp&dl=0

https://www.dropbox.com/scl/fi/np4wl08c8991z581jymtz/mnist_challenge.pkl?rlkey=p7rk54bzlr43w7s2dclegmrrv&dl=0


Each week we will upload your scores to the leaderboard.  

We will need to work out kinks in a gradual basis. I'm hoping for UFC level competition. 

