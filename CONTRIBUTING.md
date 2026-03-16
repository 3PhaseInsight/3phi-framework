# Contibuting to 3phi Framework

## Build a new version

Write & commit your code and then tag it:
```
git tag -a vx.x.x -m "vx.x.x"
git push origin vx.x.x
```

## Commit Message Guidelines

Commit Messages should follow the recommendations of [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), supplemented by the [Angular Convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines). Every commit must contain `type` and `description`, while `scope`, `body` and `footer` are optional:

```
<type>(<scope>): <description>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

#### Type
`type` is mandatory, and should be one of the following:

- `feat`: A new feature
- `fix`: A bug fix
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to our CI configuration files and scripts
- `docs`: Documentation only changes
- `perf`: A code change that improves performance
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `test`: Adding missing tests or correcting existing tests

#### Scope
`scope`is optional, but if included it should be a noun that gives context to the part of the codebase that is affected by the commit. Please do not use issue identifiers as scope.

#### Description
`description` is mandatory, and provides a short and concise description of the change. Be mindful of the following:

- use the imperative, present tense: "change" not "changed" nor "changes"
- don't capitalize the first letter
- no dot (.) at the end

#### Body
`body` is optional, but if included, it should use the imperative, present tense: "change" not "changed" nor "changes" (just as in `description`). The body should include the motivation for the change and contrast this with previous behavior.

#### Footer
`footer` is optional, but if included, it should use `-`in place of whitespace to distinguish it from a `body`or multiline `body`. It is the place to reference git issues that this commit closes.

#### Breaking Change
A commit that make incompatible API changes, should be denoted `Breaking Change`, either by a `!` right after `scope` or by adding `Breaking Change` or `Breaking-Change` to `footer`.

### Linter and formatter
[Ruff](https://docs.astral.sh/ruff/linter/) is used for linting and formatting in both local development (pre-commit) and CI, with a shared configuration in pyproject.toml.

#### Ruff in CI pipeline
- Runs on pushes/merge requests in GitLab.
- Checks only (no auto-fix). If any check fails, the pipeline fails and the change cannot be merged.

#### Ruff in pre-commit hook
- Runs locally on commit (and optionally on push).
- Applies fixes and then fails if it changed files so you can review and commit the updates.
- First-time setup:
  ```
  pre-commit install
  # optional: also run on push
  pre-commit install --hook-type pre-push
  ```
- Manual sweep of the whole repo:
  ```
  pre-commit run --all-files
  ```


## Documentation

This Project uses Sphinx for API docs. From the [docs](./docs/) run:
```
make html
```
to generate the docs. Then you can open the [docs index](./docs/_build/html/index.html) in your browser.

We leverage [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) to generate the documentation from docstrings in code.
To configure which classes are included in the API docs, edit and extend [index.rst](./docs/api/index.rst).

The currently published docs can be found [here](https://3phaseinsight.github.io/3phi-framework/index.html).
