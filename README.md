# Iris Flower Classifier

This is an introduction to machine learning in Rubix ML using the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and the [K Nearest Neighbors classifier](https://github.com/RubixML/RubixML#k-nearest-neighbors). In this tutorial, you'll learn how structure a Rubix ML project, define a learner, and train it to make predictions on a testing portion of the dataset.

- **Difficulty**: Easy
- **Training time**: Short
- **Memory needed**: < 1G

## Installation

Clone the repository locally:
```sh
$ git clone https://github.com/RubixML/Iris
```

Install dependencies:
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial
The Iris dataset consists of 50 samples from each of three species of Iris flower - Iris setosa, Iris-virginica, and Iris-versicolor. Each sample is comprised of 4 measurements (sepal length, sepal width, petal length, and petal width) which are used by the K Nearest Neighbors classifier to determine the *distance* between samples. KNN works by inferring an unknown sample's label based on its k nearest neighbors from the training set.

Before we can train the K Nearest Neighbors learner, we need to import the data from `dataset.csv` into a [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object. We'll use the League of Extraordinary PHP packages' [CSV Reader](https://csv.thephpleague.com/) to help us import the data.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

$labels = $reader->fetchColumn('class');

$dataset = Labeled::fromIterator($samples, $labels);
```

Since the data are imported as strings by default, we'll need to convert the features to their numerical counterparts so that they can be measured by the distance function. Luckily, Rubix provides a transformer that can be applied directly to the newly instantiated dataset object that will handle this for us.

```php
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
```

When training a machine learning model, it is important to set *some* of the data aside for testing purposes. By splitting the dataset into training and testing sets, we gain the ability to test the model on data it has never seen before, thus measuring its generalization abilities. Since we have discrete class labels, we can perform a *stratified* split that ensures that each subset of the dataset contains proportional counts of each label. Optionally, we randomize the dataset first to ensure that sample order does not effect the training of the learner.

```php
[$training, $testing] = $dataset->randomize()->stratifiedSplit(0.80);
```

Next we define our estimator instance with the chosen hyper-parameters. Hyper-parameters are estimator constructor parameters that influence the way the estimator learns and performs inference. K Nearest Neighbors has 2 hyper-parameters that we will consider for this tutorial - the number of nearest neighbors to consider given by *k* and the kernel distance function used to measure the distance between samples. We'll choose to use the 5 nearest neighbors and standard [Euclidean](https://github.com/RubixML/RubixML#euclidean) distance for now, but feel free to experiement with other settings. For example, you could instead choose the 3 nearest neighbors under the [Manhattan](https://github.com/RubixML/RubixML#manhattan) distance.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KNearestNeighbors(5, new Euclidean());
```

Now, we're ready to train our learner with the training set. Simply pass the dataset object to the `train()` method on the newly instantiated estimator.

```php
$estimator->train($training);
```

Once the estimator has been trained, we can use it to make predictions on the rest of the data in the testing set. To return an array of predictions, pass the testing dataset object to the `predict()` method of the estimator. 

```php
$predictions = $estimator->predict($testing);
```

We measure the performance of our model by outputting a report based on the predictions and the ground truth from the testing set. The [Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) report gives us a detailed look at how the estimator performed at classifying each sample by label.

```php
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new MulticlassBreakdown();

$results = $report->generate($predictions, $testing->labels());
```

## Original Dataset
Creator: Ronald Fisher
Contact: Michael Marshall
Email: (1) MARSHALL%PLU '@' io.arc.nasa.gov

### References
>- [1] R. A. Fisher. (1936). The use of multiple measurements in taxonomic problems.
>- [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
