# Atlas Download Tools

<p>
<a href="https://github.com/BlueBrain/Atlas-Download-Tools/blob/main/LICENSE.txt"><img src="https://img.shields.io/github/license/BlueBrain/Atlas-Download-Tools" alt="License" /></a>
&emsp;
<a href="https://github.com/BlueBrain/Atlas-Download-Tools/actions/workflows/run-tests.yml"><img src="https://github.com/BlueBrain/Atlas-Download-Tools/actions/workflows/run-tests.yml/badge.svg?branch=main" alt="Build status" /></a>
&emsp;
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black"></a>
<a href="https://pycqa.github.io/isort/"><img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="Isort"></a>
<a href="http://www.pydocstyle.org/"><img src="https://img.shields.io/badge/docstrings-pydocstyle-informational" alt="Pydocstyle"></a>
<a href="https://flake8.pycqa.org/"><img src="https://img.shields.io/badge/PEP8-flake8-informational" alt="Pydocstyle"></a>
</p>

Atlas Download Tools is a toolbox to download atlas data.

## Installation


### Installation from source
If you want to try the latest version, you can install from source.
```shell script
pip install git+https://github.com/BlueBrain/Atlas-Download-Tools
```

### Installation for development
If you want a dev install, you should install the latest version from source with
all the extra requirements for running test.
```shell script
git clone https://github.com/BlueBrain/Atlas-Download-Tools
cd Atlas-Download-Tools
pip install -e '.[dev]'
```

## How to use the package

The package `atldld` has several functionalities to download data from [Allen Brain Institute](https://portal.brain-map.org/):

- One can find dataset IDs from a gene expression name:
```python
from atldld.utils import get_experiment_list_from_gene
dataset_ids = get_experiment_list_from_gene("Pvalb", axis='sagittal')
```

- One can download any dataset from a dataset ID:
```python
from atldld.sync import download_dataset
dataset = download_dataset(dataset_id=DATASET_ID, **kwargs)
image_id, section_number, img, df = next(dataset)
```

- One can obtain metadata of a dataset:
```python
from atldld.utils import CommonQueries, get_3d
# The axis {'sagittal', 'coronal'}
axis = CommonQueries.get_axis(dataset_id=DATASET_ID)
# The reference space
ref_space = CommonQueries.get_reference_space(dataset_id=DATASET_ID)
# The 3d transformation of the dataset
matrix_3d = get_3d(dataset_id=DATASET_ID)
```

- One can download any image from an image ID and the given 2D transformation:
```python
from atldld.sync import get_transform
from atldld.utils import get_image
img = get_image(image_id=IMAGE_ID)
p, i, r = xy_to_pir_API_single(0, 0, image_id=IMAGE_ID)
# For coronal image
df = get_transform(p, dataset_id=DATASET_ID)
# For sagittal image
df = get_transform(r, dataset_id=DATASET_ID)
```

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2021 Blue Brain Project/EPFL
