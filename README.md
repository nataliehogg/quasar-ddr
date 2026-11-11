# Quasars as a self-sufficient probe of the distance duality relation

Generating mock datasets of luminosity and angular diameter distances from quasars to forecast their sensitivity to DDR violation.

## Installation

Use Python 3.10 or later. Install the dependencies with:

```bash
python -m pip install -r requirements-quasar-ddr.txt
```

## Usage

Run the notebooks from the repository root:

1. `mock_maker_interface.ipynb`: generate paired luminosity- and angular-distance mocks. Select the Current, Intermediate or Optimistic scenario in the configuration cell.
2. `sampler_interface.ipynb`: select the matching scenario or set `mock_file` to another generated `_paired_dataset.npz` file, keeping its companion `_metadata.json` alongside it. Run the sampling cells, then plot the saved chains.
3. `mrk509_eta.ipynb`: calculate the distance-duality ratio for Mrk 509.

For the paper's Figure 1, generate and sample all three scenarios using the supplied defaults, then run the comparison cells in `sampler_interface.ipynb`.
