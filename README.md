# pygen
Provides a training loop for PyTorch, plus some distribution and layer objects and callbacks for monitoring the training session.

pygen supports training either a Distribution object or a layer object. A distribution object is an object with log_prob and sample methods. A layer object is a callable which accepts a tensor and returns a Distribution object.

Distribution objects follow the PyTorch Distributions (https://pytorch.org/docs/stable/distributions.html) conventions which were in turn were based on the TensorFlow Probability Distributions package (https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution)

Layer objects follow the design principles of the TensorFlow Probability Layers package (https://www.tensorflow.org/probability/api_docs/python/tfp/layers)


## Examples

To train an mnist classifier:

```
digit_recognizer = torch.nn.Sequential(classifier_net.ClassifierNet(mnist=True), layer_categorical.Categorical())
train.LayerTrainer(digit_recognizer.to(ns.device), train_dataset).train()
```

To train an mnist PixelCNN:

```
digit_distribution = pixelcnn.PixelCNNBernoulliDistribution(event_shape=[1, 28, 28])
train.DistributionTrainer(digit_distribution.to(ns.device), train_dataset).train()
```

Full code examples:

./examples/mnist_classifier.py: Classic MNIST classifier

./examples/cifar10_classifier.py: Classic CIFAR10 classifier

./examples/mnist_pixelcnn.py: Generative MNIST model based on PixelCNN

./examples/cifar10_pixelcnn.py: Generative CIFAR10 model based on PixelCNN

## Layers

Layer objects can be thought of as representing conditional probability distributions. This could have been implemented as log_prob(y, x) representing p(x|y), but is instead implemented as a function which returns a probability distribution. The advantage is this distribution can now be passed to any method expecting a probability distribution, whereas in the former case it would require a new parallel interface to be developed.
These ideas come from the TensorFlow Probability package.

The MNIST classifier example above was built by:
```
digit_recognizer = torch.nn.Sequential(classifier_net.ClassifierNet(mnist=True), layer_categorical.Categorical())
```
but it could also have been built as:
```
class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.net = classifier_net.ClassifierNet(mnist=True)

    def forward(self):
        logits = self.net(x)
        return torch.distributions.categorical.Categorical(logits=logits)


digit_recognizer = DigitRecognizer()
```
The use of layer_categorical.Categorical() in the first approach is purely for convenience, either can be passed into a LayerTrainer. It is your personal preference as to which to use.

## Trainers

There are two trainers: train.DistributionTrainer, train.LayerTrainer

They both expect a dataset of tuples. The DistributionTrainer optimizes the distribution where the 1st element is the target. The LayerTrainer assumes the 1st element is the variable to be condition on (eg the image in the case of the MNIST dataset) and the 2nd element is the target distribution (eg the label in the case of the MNIST dataset).

## Installation and Setup

Install PyTorch (https://pytorch.org/get-started/locally/)

There is no package install currently setup, so you need to set PYTHONPATH to point to root of the repository, eg:

```
git clone https://github.com/jfrancis71/pygen.git
export PYTHONPATH=~/github_repos
```

To run the mnist classifier:

```python ./examples/mnist_classifier.py --datasets_folder=~/DataSets/ --tb_folder=./logs --device="cuda"```

where the datasets_folder option points to where you would like to store the MNIST dataset download.

The PixelCNN examples use the following repo:
```
git clone https://github.com/jfrancis71/pixel-cnn-pp.git
mv pixel-cnn-pp pixelcnn_pp
```
This is a modified fork of https://github.com/pclucas14/pixel-cnn-pp. The modifications were made to support different types of probability distributions.

## Resources

The design principles behind Distributions and Layers can be quite subtle, particularly with respect to getting the shapes correct.
So here are some resources:

PyTorch Distribution:
(https://pytorch.org/docs/stable/distributions.html)

For distribution shapes:
(https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes)

An example of Tensorflow Probability Distribution Bernoulli:
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli

An example of Tensorflow Probability Layer IndependentBernoulli:
(https://www.tensorflow.org/probability/api_docs/python/tfp/layers/IndependentBernoulli)

A discussion of PyTorch Probability Distribution shapes:
(https://bochang.me/blog/posts/pytorch-distributions/)

A general discussion on probability distribution shapes:
(https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/)
