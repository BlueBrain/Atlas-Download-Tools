# Contributing

This page includes some guidelines to enable you to contribute to the project.

## Found a bug?

If you find a bug in the source code or in using the theme, you can
open an issue on GitHub.
Even better, you can submit a pull request with a fix.

## Submission guidelines

### Submitting an issue

Before you submit an issue, please search the issue tracker, maybe an issue
for your problem already exists and the discussion might inform you of workarounds
readily available.

We want to fix all the issues as soon as possible, but before fixing a bug we
need to reproduce and confirm it. In order to reproduce bugs we will need as
much information as possible, and preferably a sample demonstrating the issue.

### Submitting a pull request (PR)

If you wish to contribute to the code base, please open a pull request by
following GitHub's guidelines.

## Development Conventions

`atldld` uses:
   - Black for formatting code
   - Flake8 for linting code
   - PyDocStyle for checking docstrings

## Generating the API documentation

If you wish to re-generate the API documentation of the package, please use the
sphinx command line as follow:

```
sphinx-apidoc src/atldld/ -Tefo docs/source/api
```
