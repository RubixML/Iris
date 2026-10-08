<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\TSNE;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new CSV('dataset.csv', header: true))
    ->apply(new FloatTypeConverter());

$embedder = new TSNE(2, 100.0, perplexity: 10, exaggeration: 6.0, epochs: 1000);

$embedder->setLogger($logger);

$dataset->apply($embedder)->exportTo(new CSV('embeddings.csv'));
