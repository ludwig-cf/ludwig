# Tests

## Unit tests

## Regression tests

The regrssion tests run the currently compiled version of the code
(with the relevant LB model) against a set of standard inputs in
the directories:
```
regression/d2q9
regression/d3q15
regresssion/d3q19-short
regression/d3q27
```

The output is then compared against a reference (`.log`) file. This test
uses three scripts from this directory:

- `test.sh`    for a given input, this checks for a test-specific tolernace
               and submits the reference `.log` file and the `.`new`
	       equivlanet to `test=diff.sh`

- `test-diff.sh`  this strips out lines which will be different but are
                  not relevant to the test, such as the time statistics,
		  and also the parallel decomposition.

- `awk-fp-diff.sh` this is a `diff`-like script which allows differences
                   in floating point numbers up to a tolerance. If this
		   exits with no output, the test has passed.

The default tolerance `1.0e-12`. This is an absolute tolerance. An
individual test input may use a different value of the tolerance by
including a comment of the form:
```
#  test_tolerance <value>
```
A common reason for specifying a reduced tolerance is that that
accumulations, such as the total energy, are not completely
robust to threads (OpenMP or GPU). If a very significantly reduced
tolerance is required, one should probably revisit the nature of
the test.
