<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use League\Csv\Reader;

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Iris Flower Classifier using K Nearest Neighbors              ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'septal-length', 'sepal-with', 'petal-length', 'petal-width',
]);

$labels = $reader->fetchColumn('class');

$dataset = Labeled::fromIterator($samples, $labels);

$dataset->apply(new NumericStringConverter());

$testing = $dataset->randomize()->take(10);

$estimator = new KNearestNeighbors(5);

$estimator->train($dataset);

$metric = new Accuracy();

$predictions = $estimator->predict($testing);

echo 'Example predictions:' . PHP_EOL;

print_r(array_slice($predictions, 0, 3));

$score = $metric->score($predictions, $testing->labels());

echo "Accuracy: $score" . PHP_EOL;

