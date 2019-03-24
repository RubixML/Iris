<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use League\Csv\Reader;

const REPORT_FILE = 'report.json';

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

[$training, $testing] = $dataset->randomize()->stratifiedSplit(0.80);

$estimator = new KNearestNeighbors(5, new Euclidean());

$estimator->train($training);

$predictions = $estimator->predict($testing);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;
