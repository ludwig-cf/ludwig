
### Contributing

Contributions may be made via a pull request. There is also a note
here on the development model, and running the tests.

Ludwig has been developed largely at The University of Edinburgh as
a collaboration between the School of Physics and Edinburgh Parallel
Computing Centre. However, contributions are welcome.

The code is released under a BSD 3-clause license.

Please note that a contribution means signing over copyright to
The University of Edinburgh.

### Running the tests

Various tests exist in the `test` subdirectory. To check that no
obvious errors have been introduced by developments one can run
```
$ cd tests
$ make
```
which runs a series of unit tests and a series of regression tests for the
LB D3Q19 model.
See the `Makefile` for further options.

Each test reports a pass or fail. Parallel tests are also available.

### Development model

The development model is borrowed from a description by Vincent Driessen
[https://nvie.com/posts/a-successful-git-branching-model/].

Two branches are always in existance: master and develop. The
master represents the current release state. New developments should be
based on the develop branch.

#### Feature branches

Create a feature branch from the develop branch
```
$ git checkout -b myfeature develop
```

When finished
```
$ git checkout develop
$ git merge --no-ff myfeature
$ git branch -d myfeature
$ git push origin develop
```

#### Patch-level updates

Fixes are intended to branch from the master and are merged back into
both master and develop when the fix is made.

```
$ git checkout -b patch-0.1.2 master
Switched to a new branch 'patch-0.1.2'
```
Make the necessary changes (with tests), increment the patch level in
version.h, and commit the changes
```
$ git commit -m "Fix such-and-such"
[patch-8-06 7dc40e0] Fix such-and-such
```

Merge the patch with the master, and then with develop
```
$ git checkout master
$ git merge --no-ff -m "Patch 0.1.2 " patch-0.1.2
Merge made by the 'recursive' strategy.
$ git checkout develop
$ git merge --no-ff -m "Patch 0.1.2" patch-0.1.2
Merge made by the 'recursive' strategy.
```

Remove the temporary branch
```
$ git branch -d patch-0.1.2
```

#### Releases

Release branches must branch from develop and must be merged back into both
develop and master.


### Code quality

One hesitates before making pronouncements on style, but here goes
anyway. As with English, the aim is not to follow rules, the aim to
be clear and unambiguous.

#### Notes on style

The code has historically been ANSI C, but the advent of GPUs has
meant that the C is a subset of ANSI which is also C++. This means
there are a few things which are legal in ANSI C, but need to be
avoided in the C++ context.

- Avoid C++ reserved words as variable names. E.g.,
  ```
  int new = 0;
  ```

- Explicit casts are required in some situations to satisfy C++. A common
  example is
  ```
  double * p = NULL;
  p = (double *) malloc(...); /* Requires the cast in C++ */
  ```

- Don't be tempted by variable-sized object initialisation, e.g.,
  ```
  int nbyte = 2;
  char tmp[nbyte+1] = {0}; /* variable sized initialisation */
  ```
  must be replaced by
  ```
  int nbyte = 2;
  char tmp[2+1] = {0};
  ```
  or some equivalent compile-time constant expression. ANSI does not
  allow variable-sized initialisation.

Finally, note we use C-style comments `/* ... */` and not C++
single line comments as a matter of stubborn consistency.


#### Avoiding CodeQL alerts

While some existing alerts remain, the goal is zero alerts. No new alerts
should be added if at all possible.

- CodeQL failures at severity "High" and above will prevent merges.
  They need to be fixed.

- Use `util_fopen()` from `util_fopen.h` in place of `fopen()`. This
  will avoid `File created without restricting permissions` errors.

- Some care can be required with integer types, e.g., something like
  ```
  {
    int b = ...;
    int c = ...;
    size_t a = c*(b+1)*sizeof(double);
  }
  ```
  will generate a `Multiplication result converted to larger type`.
  One needs to have the multiplication before the conversion.
  ```
    {
    int b = ...;
    int c = ...;
    int d = c*(b+1);
    size_t a = d*sizeof(double);
  }
```
If `c*(b+1)` may overflow, then an explicit cast to `size_t`is required.

- Anything using `scanf()` and the like must check return values.

- Try to use fixed `const char *` strings where possible, as the
  alternative of character buffers is a good source of problems.
