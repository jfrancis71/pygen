# pygen
Provides a training loop for PyTorch, plus some distribution and layer objects and callbacks for monitoring the training session.

pygen supports training either a Distribution object or a layer object. A distribution object is an object with log_prob and sample methods. A layer object is (generally) a nn.Module, but instead of accepting a tensor and returning a tensor, it accepts a tensor and returns a probability distribution.

Distribution objects follow the PyTorch Distributions (https://pytorch.org/docs/stable/distributions.html) conventions which were in turn were based on the TensorFlow Probability Distributions package (https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution)

Layer objects follow the design principles of the TensorFlow Probability Layers package (https://www.tensorflow.org/probability/api_docs/python/tfp/layers)


## Examples

To train an mnist classifier:

```
digit_recognizer = torch.nn.Sequential(classifier_net.ClassifierNet(mnist=True), layer_independent_categorical.IndependentCategorical(event_shape=[]))
train.train(classifier, train_dataset, train.layer_objective)
```


Full code examples:

./examples/classifier.py: Classic classifier (cmd line option to train either MNIST of CIFAR10)

./examples/conditional_mnist.py: Simple generative model over MNIST digits conditioned on the digit label.


## Layers

Layer objects can be thought of as representing conditional probability distributions. This could have been implemented as log_prob(y, x) representing p(x|y), but is instead implemented as a function which returns a probability distribution. The advantage is this distribution can now be passed to any method expecting a probability distribution, whereas in the former case it would require a new parallel interface to be developed.
These ideas come from the TensorFlow Probability package.

The MNIST classifier example above was built by:
```
digit_recognizer = torch.nn.Sequential(classifier_net.ClassifierNet(mnist=True),
    layer_independent_categorical.IndependentCategorical(event_shape=[]))
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
The use of layer_categorical.Categorical() in the first approach is purely for convenience. It is your personal preference as to which to use.

Take special care when using Layers in spatial tensors, i.e. tensors of form BxYxXxC. Layers, as in Tensorflow interpret the last tensor component as the parameters of the distribution. This is fine for a layer such as BxC and will work the same way in both frameworks. But note the Tensorflow format for spatial layers is BxYxXxC where layers would interpret the C components as the parameters of a probability distribution with batch shape BxYxX. Whereas the PyTorch format is BxCxYxX which you will probably want to permute to BxYxXxC. If you leave it as BxCxYxX then X will be treated as the distribution parameters with batch shape BxCxY which is probably not what you want.


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


## Resources

The design principles behind Distributions and Layers can be quite subtle, particularly with respect to getting the shapes correct.
I refer to both Tensorflow and PyTorch as both packages follow the same design principles and so reference material from either is useful.
So here are some resources:

### PyTorch

PyTorch Distribution:
(https://pytorch.org/docs/stable/distributions.html)

A discussion of PyTorch Probability Distribution shapes:
(https://bochang.me/blog/posts/pytorch-distributions/)

### Tensorflow

For distribution shapes:
(https://www.tensorflow.org/probability/examples/Understanding_TensorFlow_Distributions_Shapes)

An example of Tensorflow Probability Distribution Bernoulli:
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli

An example of Tensorflow Probability Layer IndependentBernoulli:
(https://www.tensorflow.org/probability/api_docs/python/tfp/layers/IndependentBernoulli)

Chanseok Kang notes, focussing on Tensorflow Probability Layers:
(https://goodboychan.github.io/python/coursera/tensorflow_probability/icl/2021/08/23/01-Probabilistic-layers.html)

Tensorflow Distributions:
(https://arxiv.org/pdf/1711.10604)

### General

A general discussion on probability distribution shapes:
(https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/)
