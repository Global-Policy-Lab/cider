# cider
## poverty prediction and targeting with mobile phone metadata

### Documentation
Visit [cider's documentation](https://global-policy-lab.github.io/cider-documentation/intro.html). 

### Deployment
To install, and manage dependencies and virtual environments this project uses Poetry. Follow the [instructions](https://python-poetry.org/docs/) to
install Poetry.

From the root directory `poetry update` followed by `poetry install`, this will establish a venv with all the needed dependencies.

Once your venv is made you can use `poetry run [command]` to run a single CLI command inside the venv.

You can use `poetry shell` to enter into the venv.

Note: On Windows, you may need to install the [numpy+mkl wheel](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy). Download the file, then run `pip install filename.whl` in the directory.

### Helper Functions
To support some helper functions that are portable across operating systems we use make. There are many implementations of this functionality for all
operating systems. Once you have downloaded one that suits you and setup poetry you can run:

* `make test [paths]` to run all pytests
* `make lint [paths]` to lint and apply changes automatically
* `make lint-dryrun [paths]` to lint the code but only print warnings
* `make clear-nb` to clear the results out of notebooks before committing them back to the repo. This helps avoid bloat from binary blobs, and keeps the changes to notebooks readable in diff tools.

### Contributing
Before contributing code please:

* Run `make clear-nb` if you have made any changes to Jupyter notebooks you would like to commit.
* Run `make lint [paths]` and manually correct any errors in the files you are changing.
* Run `poetry update` if you made any changes to the dependencies. This will regenerate the `poetry.lock` file.
* Run `make_test` and verify that the tests still pass. If they fail, confirm if they fail on master before assuming your code broke them.


### Testing
For testing we use `pytest`. Some guidelines:

* In any directory with source code there should be a `tests` folder that contains files that begin with `test_` e.g. `test_foo_bar_file.py`.
* Within each test file each function that is a test should start with the word `test`, in source code no function should start with the word `test_`.
* We should attempt to write unit tests wherever possible. These are minimal tests that confirm the functionality of one layer of abstraction. We should use the `unittest.mock` standard python library to make mock objects if one layer of unit tests requires interaction with objects from a different layer of abstraction. This ensures the tests are fast, and it decouples the pieces of code making test failures more meaningful as the failure will likely be contained in unit tests whereas one failure would propigate to cause cascading failures in integration tests.i
* We can write integration or smoke tests which attempt to run the code end-to-end. These should not be exhaustive and all such tests should take less than a few minutes to run total.
* Developers should be familiar with the `pytest` concepts of `fixtures` to make the setup for tests repeatable, `parametrize` to make a large number of variations on the same test, and `pytest.raises` to check that the correct type of errors are thrown when they should be.

## License
Copyright ©2022-2023. The Regents of the University of California (Regents). All Rights Reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.