# BRAILS Documentation

This is the repository where the documentation for BRAILS is maintained. The [current documentation](https://nheri-simcenter.github.io/BRAILS-Documentation/index.html) is in reStructuredText format and is built using the Sphinx Python module.

Contributors should follow the [style reference](https://github.com/NHERI-SimCenter/SimCenterDocumentation/blob/master/Help/docstyle.md) for guidelines on documentation formatting.

## Directory Structure

+ `docs/`   - the folder containing the most up to date material (all other directories contain legacy material)
+ `docs/Makefile` - Legacy Linux and MacOS makefile for building current document
+ `docs/make.bat`   - Legacy Windows make.bat to do same thing
+ `docs/conf.py` - configuration file (in this file set app for which doc is requiretd)
+ `docs/index.rst` - main index file, which pulls in files in `docs/common`

## Building the Documentation

Documentation files for BRAILS can be built by completing the following steps.

### 1. Download this repository from GitHub

For Git users, this can be done by running the following command in a terminal.

```shell
git clone https://github.com/NHERI-SimCenter/BRAILS.git
```

The remaining terminal commands should be run from the `docs/` directory of this repository, herein referred to as the *documentation root*.

### 2. Install dependencies

Install the project dependencies by running the following terminal command from the *documentation root*:

```shell
pip install -r requirements.txt
```

or

```shell
pip3 install -r requirements.txt
```

> Note: A Python 3 installation must be available in your terminal environment. The pip command is used on Windows and pip3 on a Mac. If you do not have admin rights, add a -U before the -r.

### 3 Build with Make

On systems with `make` installed, the following command can be run from the documentation root to build a particular documentation target. `make` is typically available by default for MacOS and Linux users and can readily be installed on Windows. Further instructions for Windows will soon be provided.

```shell
make <target>...
```

where `<target>` is one of:

| `<target>` | description |
|------------|-------------|
|  `web`    | Generate HTML output in the app publishing repository (i.e., `../<app-name>-Documentation/`).
|  `html`   | Generate HTML output in `build/<app-name>/html/`.
|  `latex`  | Generate LaTeX output in `build/<app-name>/pdf/`.
|  `pdf`    | Generate PDF output in `build/<app-name>/pdf/`.


Several targets may be chained at the end of a command for a particular application, as shown in the [examples](#examples) below.

### 4. Examples

- The following command will generate **HTML** output for the BRAILS documentation in the directory `docs/web/html/`:

    ```shell
    make html
    ```

- The following command will generate **latex** and **pdf** output for the BRAILS documentation in the directories `docs/web/latex/`, and `docs/web/pdf/`, respectively:

    ```shell
    make latex pdf
    ```
    Note, however, that in order to achieve a proper build, one may need to run the `make latex` target several times in succession before running `make pdf`.

> Note: Legacy build scripts in the `docs` directory do not sync example files from their source repositories.

<!--
If pdf can not be built, try this in the latex file:
```
\usepackage[backend=biber]{biblatex}
```
-->
