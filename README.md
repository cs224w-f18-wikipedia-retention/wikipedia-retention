wikipedia-retention
==============================

Graph analysis of Wikipedia edit history.

This project follows the [Cookiecutter Data Science](Cookiecutter Data Science) conventions.

## Quickstart

```bash
$ python setup.py bdist_egg
$ spark-submit \
    --driver-memory 8096m \
    --py-files dist/*.egg \
    runner.py --help
```

## Reference

```bash
spark-submit \
    --conf spark.driver.memory=16000m \
    --conf spark.local.dir=`realpath ~/tmp` \
    --py-files dist/src-0.1.0-py3.6.egg \
        runner.py bz2parquet \
            --input-path data/raw/enwiki-20080103.main.bz2 \
            --output-path data/interim/enwiki-meta-main
```