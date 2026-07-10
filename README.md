# Interpretable Strategies for Yahtzee: An Imitation Learning Approach

Code and notebooks supporting the Master's Thesis "Interpretable Strategies for Yahtzee: An Imitation Learning Approach".

## Repository Contents
 
| Path | Description |
|---|---|
| `thesis_paper_SzymonKacprzyk.pdf` | Full thesis: "Interpretable Strategies for Yahtzee: An Imitation Learning Approach" |
| `implementation.ipynb` | Main notebook: model training and experiments |
| `data analysis.ipynb` | Evaluation, metrics, and plots used in the thesis |
| `generation.py` | Generates simulated Yahtzee gameplay data |
| `dagger_execution.py` | Runs the DAgger data-aggregation / imitation learning loop |
| `Extracted trees and decision paths/` | Saved decision trees and extracted decision paths |
| `Images/` | Figures and plots used in the thesis and notebooks |

## Setup
 
This project depends on code from a companion repository. Before running anything, clone both repositories into the **same working directory**:
 
```bash
git clone https://github.com/skacprzyk8/interpretable_yahtzee.git
git clone https://github.com/papetronics/case-studies-final-project.git
```
 
Make sure the contents of `case-studies-final-project` sit alongside the files in this repository (i.e. in the same folder), since the notebooks and scripts import from it directly.
 
> **Note:** Some files use hardcoded local paths. Update these paths to match your local environment before running the notebooks or scripts.

For full details and results, see the thesis associated with this work also included in this repository.
