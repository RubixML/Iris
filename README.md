# Rubix ML - Iris Flower Classifier
This is a lightweight introduction to machine learning in Rubix ML using the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and the K Nearest Neighbors algorithm. In this tutorial, you'll learn how structure a project, instantiate a learner, and train it to make predictions on some testing data.

- **Difficulty**: Easy
- **Training time**: Seconds
- **Memory needed**: < 1G

## Installation
Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Iris
```

Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial

### Introduction
The Iris dataset consists of 50 samples from each of three species of Iris flower - Iris setosa, Iris virginica, and Iris versicolor (pictured below). Each sample is comprised of 4 measurments or *features* - sepal length, sepal width, petal length, and petal width. Our objective is to train a [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) (KNN) classifier to determine the species of Iris flower from a set of unknown samples. KNN is an intuitive algorithm that is easy to understand for most people. Let's get started!

![Iris Flower Species](https://raw.githubusercontent.com/RubixML/Iris/master/docs/images/iris-species.png)

### Extracting the Data
Before we can train the learner, we must import the data from the `dataset.csv` file into our project. We'll use the League of Extraordinary PHP packages' [CSV Reader](https://csv.thephpleague.com/) to help us import the data from the CSV (comma-separated values) file.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Iris/blob/master/train.php) file in project root.

```php
use League\Csv\Reader;

$reader = Reader::createFromPath('dataset.csv')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

$labels = $reader->fetchColumn('class');
```

The return values of the `getRecords()` and `fetchColumn()` methods are iterators which we'll load into a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object using the `fromIterator()` static factory method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

### Dataset Preparation
Since the data from the CSV file are imported as strings by default, we'll need to convert them to their numerical representations before proceeding. Luckily, the [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html) can be applied to the newly instantiated dataset object. Numeric String Converter is an example of a dataset [Transformer](https://docs.rubixml.com/en/latest/transformers/api.html) because it modifies the dataset in some way. To apply the transformation, pass the transformer instance to the `apply()` method on the dataset object.

```php
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
```

In addition, we'll set aside 10 random samples that we'll later use to make some example predictions and score the model that we train. The `randomize()` method on the dataset object will handle shuffling the data while `take()` pulls the first n rows from the dataset and puts them into a separate dataset object.

```php
$testing = $dataset->randomize()->take(10);
```

### Instantiating the Learner
Next, we'll instantiate the [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) classifier and choose the value of the *k* hyper-parameter. Hyper-parameters are constructor parameters that effect the behavior of the learner during training and inference. KNN is a distance-based algorithm that finds the k closest samples from the training set and takes the most frequent label as the prediction. For example, if we choose k equal to 5, then we may get 4 labels that are Iris setosa and 1 that is Iris virginica. In this case, the estimator would predict Iris-setosa because that is the most common label. To instantiate the learner, pass the hyper-paremeter k into the constructor of KNN. Refer to the docs for more on KNN's additional hyper-parameters.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new KNearestNeighbors(5);
```

### Training
Now, we're ready to train the learner by calling the `train()` method with the training set we instantiated earlier.

```php
$estimator->train($dataset);
```

### Making Predictions
With the model trained, we can make predictions using the testing data by calling the `predict()` method with the testing set.

```php
$predictions = $estimator->predict($testing);
```

During inference, the KNN algorithm interprets the features of the samples as spatial coordinates and uses the *distance* between samples to determine the most similar samples from the data it has already seen. From the visualization below, the features of each species of Iris flower form distinct clusters that can be learned by the K Nearest Neighbors algorithm.

![Iris Dataset 3D Plot](https://raw.githubusercontent.com/RubixML/Iris/master/docs/images/iris-dataset-3d-plot.png)

### Validation Score
We can test the model generated during training by comparing the predictions it makes to the ground-truth labels from the testing set. We'll need to choose a cross validation [Metric](https://docs.rubixml.com/en/latest/cross-validation/metrics/api.html) to output a score that we'll interpret as the generalization ability of our newly trained estimator. The [Accuracy](https://docs.rubixml.com/en/latest/cross-validation/metrics/accuracy.html) is a simple classification metric that ranges from 0 to 1 and is calculated as the number of correct predictions to the total number of predictions. To obtain the validation score, pass the predictions we generated from the model earlier along with the labels from the testing set to the `score` method on the metric instance.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy is $score" . PHP_EOL;
```

**Output**

```sh
Accuracy is 0.9
```

Nice work!

### Next Steps
Congratulations on completing the introduction to machine learning in PHP with Rubix ML. Now you're ready to experiment on your own. For example, you may want to try different values of the *k* hyper-parameter or swap out the default [Euclidean](https://docs.rubixml.com/en/latest/kernels/distance/euclidean.html) distance kernel for either [Manhattan](https://docs.rubixml.com/en/latest/kernels/distance/manhattan.html) or [Minkowski](https://docs.rubixml.com/en/latest/kernels/distance/minkowski.html). Information on all of the [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) hyper-parameters can be found in the docs.

## Original Dataset
Creator: Ronald Fisher
Contact: Michael Marshall
Email: (1) MARSHALL%PLU '@' io.arc.nasa.gov

### References
>- R. A. Fisher. (1936). The use of multiple measurements in taxonomic problems.
>- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
