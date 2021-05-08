<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$logger = new Screen();

$logger->info('Loading data into memory');

$training = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$testing = $training->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

$logger->info('Training');

$estimator->train($training);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$metric = new Accuracy();

$score = $metric->score($predictions, $testing->labels());

$logger->info("Accuracy is $score");
