<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\TruncatedSVD;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new NDJSON('dataset.ndjson'));

$stats = $dataset->describe();

echo $stats;

$stats->toJSON()->saveTo(new Filesystem('stats.json'));

$logger->info('Stats saved to stats.json');

$dataset->apply(new TruncatedSVD(2))
    ->exportTo(new CSV('embedding.csv'));

$logger->info('Embedding saved to embedding.csv');
