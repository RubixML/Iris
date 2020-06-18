# Rubix ML - Iris Flower Classifier
A lightweight introduction to machine learning in Rubix ML using the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and the K Nearest Neighbors algorithm. By the end of this tutorial, you'll know how to structure a project, instantiate a learner, and train it to make predictions on some test data.

- **Difficulty**: Easy
- **Training time**: Less than a minute

## Installation
Clone the project locally using [Composer](https://getcomposer.org/):
```sh
$ composer create-project rubix/iris
```

## Requirements
- [PHP](https://php.net) 7.2 or above

## Tutorial

### Introduction
The Iris dataset consists of 50 samples for each of three species of Iris flower - Iris setosa, Iris virginica, and Iris versicolor (pictured below). Each sample is comprised of 4 measurements or *features* - sepal length, sepal width, petal length, and petal width. Our objective is to train a [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) (KNN) classifier to determine the species of Iris flower from a set of unknown test samples using the Iris dataset. Let's get started!

![Iris Flower Species](https://raw.githubusercontent.com/RubixML/Iris/master/docs/images/iris-species.png)

### Extracting the Data
The first step is to extract the Iris dataset from the `dataset.ndjson` file in our project folder into our training script. You'll notice that we've provided the Iris dataset in CSV (Comma-separated Values) format as well. This is strictly for convenience in case you wanted to view the dataset in your favorite spreadsheet software. To instantiate a new [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object we'll pass an [NDJSON](https://docs.rubixml.com/en/latest/extractors/ndjson.html) extractor pointing to the dataset file in our project folder to the `fromIterator()` factory method. The factory uses the last column of the data table for the labels and the rest of the columns for the values of the sample features. We'll call this our *training* set.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Iris/blob/master/train.php) file in project root.

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;

$training = Labeled::fromIterator(new NDJSON('dataset.ndjson'));
```

Next, we'll set aside 10 random samples that we'll use later to make some example predictions and score the model. The `randomize()` method on the dataset object will handle shuffling the data to ensure randomness and the `take()` method pulls the first *n* rows from the training set and puts them into a separate dataset object. We do this because we want to test the model on samples that it hasn't been trained with.

```php
$testing = $dataset->randomize()->take(10);
```

### Instantiating the Learner
Next, we'll instantiate the [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) classifier and choose the value of the `k` hyper-parameter. Hyper-parameters are constructor parameters that effect the behavior of the learner during training and inference. KNN is a distance-based algorithm that finds the *k* closest samples from the training set and predicts the label that is most common. For example, if we choose `k` equal to 5, then we may get 4 labels that are `Iris setosa` and 1 that is `Iris virginica`. In this case, the estimator would predict Iris-setosa because that is the most common label. To instantiate the learner, pass the value of hyper-parameter `k` to the constructor of the learner. Refer to the docs for more info on KNN's additional hyper-parameters.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new KNearestNeighbors(5);
```

### Training
Now, we're ready to train the learner by calling the `train()` method with the training set we prepared earlier.

```php
$estimator->train($training);
```

### Making Predictions
With the model trained, we can make predictions using the testing data by calling the `predict()` method on the testing set.

```php
$predictions = $estimator->predict($testing);
```

During inference, the KNN algorithm interprets the features of the samples as spatial coordinates and uses the *distance* between samples to determine the most similar samples from the data it has already seen. From the visualization below, the features of each species of Iris flower form distinct clusters that can be learned by the K Nearest Neighbors algorithm.

![Iris Dataset 3D Plot](https://raw.githubusercontent.com/RubixML/Iris/master/docs/images/iris-dataset-3d-plot.png)

### Validation Score
We can test the model generated during training by comparing the predictions it makes to the ground-truth labels from the testing set. We'll need to choose a cross validation [Metric](https://docs.rubixml.com/en/latest/cross-validation/metrics/api.html) to output a score that we'll interpret as the generalization ability of our newly trained estimator. The [Accuracy](https://docs.rubixml.com/en/latest/cross-validation/metrics/accuracy.html) is a simple classification metric that ranges from 0 to 1 and is calculated as the number of correct predictions to the total number of predictions. To obtain the accuracy score, pass the predictions we generated from the model earlier along with the labels from the testing set to the `score` method on the metric instance.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo 'Accuracy is ' . (string) ($score * 100.0) . '%' . PHP_EOL;
```

Now you're ready to run the training script from the command line.
```sh
php train.php
```

**Output**

```sh
Accuracy is 90%
```

### Next Steps
Congratulations on completing the introduction to machine learning in PHP with Rubix ML using the Iris dataset. Now you're ready to experiment on your own. For example, you may want to try different values of `k` or swap out the default [Euclidean](https://docs.rubixml.com/en/latest/kernels/distance/euclidean.html) distance kernel for another one such as [Manhattan](https://docs.rubixml.com/en/latest/kernels/distance/manhattan.html) or [Minkowski](https://docs.rubixml.com/en/latest/kernels/distance/minkowski.html).

## Original Dataset
Creator: Ronald Fisher
Contact: Michael Marshall
Email: (1) MARSHALL%PLU '@' io.arc.nasa.gov

### References
>- R. A. Fisher. (1936). The use of multiple measurements in taxonomic problems.
>- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## License
The code is licensed [Apache 2.0](LICENSE.md) and the tutorial is licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
