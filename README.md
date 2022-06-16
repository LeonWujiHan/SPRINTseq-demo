# SPRINTseq-demo

## Overview

This repository contains the code for visualizing SPRINTseq spatial datasets. The datasets can be inquired individually.

## Installation

1. It is recommended that you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. 

2. Clone this repo, and create the provided [environment](./environment.yaml):

```
$ git clone https://github.com/LeonWujiHan/SPRINTseq-demo.git
$ cd SPRINTseq-demo
$ conda env create -f environment.yaml
```

3. Activate the environment, and install package dependencies:

```
$ conda activate SPRINTseq-demo
$ pip install -r requirements.txt
```

## Basic Usage

To run the scripts, simply copy the compressed data archive to the repo root directory, and extract the files:

```
$ tar -xzvf data.tar.gz
```

You should see two table files (`cells.csv` and `rna_labeled.csv`) in the `data` subfolder. Then run the script:

```
$ streamlit run View_Cells.py
```

## Questions

Please reach out to [Leon](mailto:leonwujihan@outlook.com) if you have any questions running the scripts, or [open an issue](../../issues/new) on GitHub.

## License

We provide this open source software without any warranty under the [MIT license](https://opensource.org/licenses/MIT).