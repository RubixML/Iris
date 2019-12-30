<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

echo 'Loading data into memory ...' . PHP_EOL;

$training = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$testing = $training->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

echo 'Training ...' . PHP_EOL;

$estimator->train($training);

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($testing);

echo 'Example predictions:' . PHP_EOL;

print_r(array_slice($predictions, 0, 3));

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy is $score" . PHP_EOL;
