# Code Base Investigator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5018973.svg)](https://doi.org/10.5281/zenodo.5018973)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8679/badge)](https://www.bestpractices.dev/projects/8679)

Code Base Investigator (CBI) is an analysis tool that provides insight into the
portability and maintainability of an application's source code.

- Measure [code divergence](http://doi.org/10.1109/P3HPC.2018.00006) and
  platform coverage to understand how much code is specialized for different
  compilers, operating systems, hardware micro-architectures and more.

- Visualize the distance between the code paths used to support different
  compilation targets.

- Identify stale, legacy, code paths that are unused by any compilation target.

- Export metrics and code path information required for P3 analysis using [other
  tools](https://p3hpc.github.io/p3-analysis-library/).


## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contribute](#contribute)
- [License](#license)
- [Security](#security)
- [Code of Conduct](#code-of-conduct)
- [Citations](#citations)


## Dependencies

- jsonschema
- Matplotlib
- NumPy
- pathspec
- Python 3
- SciPy
- tabulate
- tqdm


## Installation

The latest release of CBI is version 2.0.0. To download and install this
release, run the following:

```
pip install git+https://github.com/P3HPC/code-base-investigator@2.0.0
```

We strongly recommend installing CBI within a [virtual
environment](https://docs.python.org/3/library/venv.html).

## Getting Started

After installation, run `codebasin -h` to see a complete list of options.

A full tutorial can be found in the [online
documentation](https://p3hpc.github.io/code-base-investigator/).


## Contribute

Contributions to CBI are welcome in the form of issues and pull requests.

See [CONTRIBUTING](CONTRIBUTING.md) for more information.


## License

[BSD 3-Clause](./LICENSE)


## Security

See [SECURITY](SECURITY.md) for more information.

The main branch of CBI is the development branch, and should not be used in
production.  Tagged releases are available
[here](https://github.com/P3HPC/code-base-investigator/releases).


## Code of Conduct

Intel has adopted the Contributor Covenant as the Code of Conduct for all of
its open source projects. See [CODE OF CONDUCT](CODE_OF_CONDUCT.md) for more
information.


## Citations

If your use of CBI results in a research publication, please consider citing
the software and/or the papers that inspired its functionality (as
appropriate). See [CITATION](CITATION.cff) for more information.
