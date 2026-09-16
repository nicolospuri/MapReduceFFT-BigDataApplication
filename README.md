# MapReduce Fair FFT

This project implements a distributed, fair version of the Farthest-First
Traversal (FFT) algorithm using Apache Spark and PySpark. It was developed for
the Big Data Computing Homework 1 assignment.

## Problem

Given a set of multidimensional points divided into two groups, `A` and `B`,
the goal is to choose a fixed number of representative centers from each
group. The requested numbers are:

- `Ka`: centers that must be selected from group `A`
- `Kb`: centers that must be selected from group `B`

The selected centers are evaluated with the clustering objective function: the
maximum Euclidean distance between any input point and its closest center.
This fairness constraint prevents the result from being determined only by
the largest or most geometrically spread group.

## Solution

The implementation in `G48HW1.py` uses a two-round MapReduce strategy:

1. The CSV data is loaded into an RDD and randomly repartitioned into `L`
   Spark partitions.
2. Each partition runs Fair FFT locally, selecting farthest points while
   respecting local quotas derived from `Ka`, `Kb`, and the numbers of `A` and
   `B` points in the complete dataset.
3. The local centers are collected as a coreset.
4. Fair FFT runs again on the coreset to produce the final `Ka + Kb` centers.
5. The program computes the maximum distance from every input point to its
   nearest final center and prints the objective value.

For the farthest-first selection, the first center is chosen randomly from a
group with a remaining quota. Subsequent centers are selected by repeatedly
choosing the point with the greatest distance to its nearest already-selected
center, skipping points from groups whose quotas are already full.

## Input Format

The input is a comma-separated file with no header. Each row contains any
number of floating-point coordinates followed by a group label:

```text
1.0,2.0,B
8.5,1.5,A
```

All rows must have the same number of coordinates, and labels must be `A` or
`B`. The program accepts local paths as well as paths readable by Spark, such
as an HDFS path.

## Usage

Run the program with Spark's submit command:

```bash
spark-submit G48HW1.py <data_path> <Ka> <Kb> <L>
```

Arguments:

- `<data_path>`: input CSV path
- `<Ka>`: requested number of centers from group `A`
- `<Kb>`: requested number of centers from group `B`
- `<L>`: number of Spark partitions

`Ka` and `Kb` must not exceed the number of points available in their groups.
`L` must be a positive integer and must not exceed the total number of points.
At least one of `Ka` or `Kb` should be greater than zero.

### Examples

```bash
spark-submit G48HW1.py testinputN32D2.csv 2 2 4
spark-submit G48HW1.py testinputN72D4.csv 2 2 4
spark-submit G48HW1.py uber_small.csv 5 5 8
```

The first execution can be run locally with a single Spark worker using:

```bash
spark-submit --master local[*] G48HW1.py testinputN32D2.csv 2 2 4
```

## Output

The program prints:

- the parsed command-line arguments;
- the total number of points and the number in each group;
- each final center and its group label;
- `Objective function`, the maximum nearest-center distance;
- `Running time of MRFairFFT`, measured in milliseconds for the two-round
  center-selection phase.

The first-center selection uses NumPy's random number generator, so center
selection and the reported objective can vary between executions unless the
random seed is controlled externally.

## Requirements

- Python 3
- Apache Spark with `spark-submit`
- PySpark
- NumPy

Install the Python packages in an environment where Spark is available:

```bash
python3 -m pip install pyspark numpy
```

Verify the installation with:

```bash
python3 -m py_compile G48HW1.py
```

## Repository Contents

- `G48HW1.py`: PySpark implementation of Fair FFT and MRFairFFT.
- `testinputN32D2.csv`: small two-dimensional test dataset.
- `testinputN72D4.csv`: small four-dimensional test dataset.
- `uber_small.csv`: larger two-dimensional test dataset.