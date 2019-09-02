# Iris Flower Classifier
This is a lightweight introduction to machine learning in Rubix ML using the famous [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and the [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) algorithm. In this tutorial, you'll learn how structure a  project, instantiate a learner, and train it to make predictions on a testing portion of the dataset.

- **Difficulty**: Easy
- **Training time**: Seconds
- **Memory needed**: 1G

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
The Iris dataset consists of 50 samples from each of three species of Iris flower - Iris setosa, Iris-virginica, and Iris-versicolor. Each sample is comprised of 4 measurments or *features* - sepal length, sepal width, petal length, and petal width. Our objective is to train a K Nearest Neighbors classifier to determine the species of a set of unknown samples. KNN is an intuitive algorithm that is easy to understand for most beginners. Let's get started!

![Iris Species](https://raw.githubusercontent.com/RubixML/Iris/master/docs/images/iris-species.png)

### Extracting the Data
Before we can train the K Nearest Neighbors learner, we need to import the data from `dataset.csv` into a dataset object. We'll use the League of Extraordinary PHP packages' [CSV Reader](https://csv.thephpleague.com/) to help us import the data. The return values of the `getRecords()` and `fetchColumn()` methods are iterators which we'll load into a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset.

> **Note:** The source code for this example can be found in the [train.php](https://github.com/RubixML/Iris/blob/master/train.php) file in project root.

```php
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

$labels = $reader->fetchColumn('class');
```

### Dataset Preparation
Then load the samples and labels into a Labeled dataset object by passing them to the `fromIterator()` static factory method.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::fromIterator($samples, $labels);
```

Since the data from CSV are imported as string types by default, we'll need to convert those features to their numerical representations before proceeding. Luckily, Rubix ML provides the [Numeric String Converter](https://docs.rubixml.com/en/latest/transformers/numeric-string-converter.html) transformer that can be applied directly to the newly instantiated dataset object that will handle this for us.

```php
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
```

Before we train the learner, we'll set 10 random samples aside that we'll later use to make some example predictions and score the model. The `randomize()` method on the dataset object will handle shuffling the data while `take()` pulls the first n rows from the dataset and puts them into a separate dataset object. For this example, we'll use 10 of the 150 samples for our testing set.

```php
$testing = $dataset->randomize()->take(10);
```

### Instantiating the Learner
Next we'll instantiate the [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) classifier instance. KNN is a distance-based algorithm that finds the k closest samples from the training set and takes the most frequent label as the prediction. For example, if we choose k equal to 5, then we may get 4 labels that are Iris-setosa and 1 labeled Iris-virginica. In this case, the estimator would predict Iris-setosa with 80% certainty.

```php
use Rubix\ML\Classifiers\KNearestNeighbors;

$estimator = new KNearestNeighbors(5);
```

The constructor parameter k is an example of a *hyper-parameter*. Hyper-parameters are constructor parameters that effect the behavior of the learner during training and inference. Refer to the documentation for more information on KNN's additional hyper-parameters.

### Training
Now we're ready to train the learner with the training set by calling the `train()` method on the learner instance.

```php
$estimator->train($dataset);
```

### Inference
Then make the predictions on the testing portion of the dataset that we set aside earlier by calling `predict()`.

```php
$predictions = $estimator->predict($testing);
```

### Validation Score
Lastly, we can test the model by comparing the predictions to the ground truth labels from the testing set. We'll use the [Accuracy](https://docs.rubixml.com/en/latest/cross-validation/metrics/accuracy.html) metric to output a score that we'll use to interpret the generalization ability of our newly trained estimator.

```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy: $score" . PHP_EOL;
```

**Output**

```sh
Accuracy: 0.9
```

### Wrap Up
- Data can be extracted from a number of sources including CSV files and databases.
- Samples and labels are passed in Rubix ML using specialized containers called [Dataset](https://docs.rubixml.com/en/latest/datasets/api.html) objects.
- The [K Nearest Neighbors](https://docs.rubixml.com/en/latest/classifiers/k-nearest-neighbors.html) algorithm searches for the k closest samples from the training set and predicts the most frequent label.
- A hyper-parameter is a setting that alters the behavior of a learning algorithm such as k in the KNN learner.
- It is important to test the learner on a separate testing set if you want to test its generalization ability.

### Next Steps
Congratulations on completing the introduction to machine learning in PHP with Rubix ML. Now you're ready to experiment on your own. We highly recommend browsing the [docs](https://docs.rubixml.com/en/latest/) and taking a look at the tutorials and example projects on our [GitHub page](https://github.com/RubixML). Feel free to post a question if you need help.

## Original Dataset
Creator: Ronald Fisher
Contact: Michael Marshall
Email: (1) MARSHALL%PLU '@' io.arc.nasa.gov

### References
>- [1] R. A. Fisher. (1936). The use of multiple measurements in taxonomic problems.
>- [2] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
