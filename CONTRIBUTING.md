# How to Contribute

Thank you for your interest in contributing to Code Base Investigator! We
welcome community contributions. You can:

- Submit your changes directly with a [pull request][1].
- File a bug or open a feature request with an [issue][2].

[1]: https://github.com/P3HPC/code-base-investigator/pulls
[2]: https://github.com/P3HPC/code-base-investigator/issues

# Pull Requests

This project follows the [GitHub flow][3]. To submit your change directly to
the repository:

- Fork the repository and develop your patch.
- Make sure your code is in line with our [coding conventions][4].
- Consider adding a [test][5].
- Submit a [pull request][6] into the main branch.

[3]: https://guides.github.com/introduction/flow/index.html
[4]: #coding-conventions
[5]: #testing
[6]: https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request

# Dependencies and Hooks

We recommend installing the following additional dependencies:

- pre-commit

These packages will allow the use of several development hooks, designed to
automatically enforce the project's coding conventions.

To install these dependencies and configure the hooks, run:

    $ pip install .
    $ pre-commit install

# Coding Conventions

For code:
- Follow PEP8 for Python code, with a line limit of 79 characters.

For commits:
- Limit the first line of Git commit messages to 50 characters.
- Limit the other lines of Git commit messages to 72 characters.
- Please consider following the guidelines outlined [here][7].

[7]: https://cbea.ms/git-commit/

# Testing

Code Base Investigator uses the Python [unittest][8] unit testing framework.
If you are contributing a new feature or bug fix, please also consider
providing an associated test. Pull requests with tests are more likely to be
accepted quickly.

Existing tests and information about how to run them can be found in the
[tests](tests) directory.

[8]: https://docs.python.org/3/library/unittest.html

# License

Code Base Investigator is licensed under the terms in [LICENSE](LICENSE). By
contributing to the project, you agree to the license and copyright terms
therein and release your contribution under these terms.

# Sign Your Work

Please use the sign-off line at the end of the patch. Your signature certifies
that you wrote the patch or otherwise have the right to pass it on as an
open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Your Name <your.name@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.
