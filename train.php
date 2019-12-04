<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use League\Csv\Reader;

echo 'Loading data into memory ...' . PHP_EOL;

$reader = Reader::createFromPath('dataset.csv')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

$labels = $reader->fetchColumn('class');

$dataset = Labeled::fromIterator($samples, $labels);

$dataset->apply(new NumericStringConverter());

$testing = $dataset->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

echo 'Training ...' . PHP_EOL;

$estimator->train($dataset);

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($testing);

echo 'Example predictions:' . PHP_EOL;

print_r(array_slice($predictions, 0, 3));

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy is $score" . PHP_EOL;

