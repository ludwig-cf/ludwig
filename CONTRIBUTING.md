
### Contributing

Please consider pull request "thorugh the usual channels".

### Tests

Tests may be run as follows:
```
$ cd tests
$ make compile-serial-d3q19
$ make run-serial-unit
$ make run-serial-regr-d3q19
```
Each test reports a pass or fail.


### Development model

The development model is borrowed from a description by Vincent Driessen
[https://nvie.com/posts/a-successful-git-branching-model/]

Two branches are always in existance: master and develop.

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

