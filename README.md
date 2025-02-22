# OrgaCCC
 Orthogonal graph autoencoders for constructing cell-cell communication networks on spatial transcriptomics data

## Getting Started

### Dependencies and requirements
OrgaCCC depends on the following packages: matplotlib, numpy, pandas, scanpy, scipy, tensorflow. See dependency versions in `requirements.txt`.Installation of the dependencies may take several minutes.
```
pip install --requirement requirements.txt
```
### Input Data
The input data should be an `AnnData` object (`adata`) with the following structure:
- **Gene Expression Matrix**: Cells in rows and genes in columns.
- **`adata.obs['celltype']`**: A column containing cell type annotations for each cell.
- **`adata.obsm['spatial']`**: A two-column array storing the spatial coordinates `(x, y)` of each cell.

Example:
```
adata = sc.read_h5ad(r"data/cortex.h5ad")
```
We provide a preprocessing module to transform the spatial coordinates of single-cell transcriptomics data into an adjacency matrix.




