# OrgaCCC
 Orthogonal graph autoencoders for constructing cell-cell communication networks on spatial transcriptomics data.
![overview](https://github.com/user-attachments/assets/1bdea92e-25a5-4406-a222-f8e1639e3c44)

 
OrgaCCC reconstructs **cell-cell and cell-type communication networks** using the orthogonal graph autoencoders at both the cell/spot and gene levels. Through **clustering analysis**, it refines cell classification and uncovers developmental trajectories, as well as spatial and functional heterogeneity of cells. Additionally, OrgaCCC employs **gene sensitivity analysis** to identify key genes involved in cell communication, offering valuable insights into critical molecular interactions.

## Getting Started

### System Requirements
OrgaCCC is implemented in **python 3.11.3**.

### Dependencies and requirements
OrgaCCC depends on the following packages: matplotlib, numpy, pandas, scanpy, scipy, tensorflow. See dependency versions in `requirements.txt`.Installation of the dependencies may take several minutes.
```
pip install --requirement requirements.txt
```
### Usage

The input data should be an `AnnData` object (`adata`) with the following structure:
- **Gene Expression Matrix**: Cells in rows and genes in columns.
- **`adata.obs['celltype']`**: A column containing cell type annotations for each cell.
- **`adata.obsm['spatial']`**: A two-column array storing the spatial coordinates `(x, y)` of each cell.

Example:
```
adata = sc.read_h5ad(r"data/cortex.h5ad")
```
 If your file is located elsewhere, you will need to manually update the path in `main.py`
We provide a preprocessing module to transform the spatial coordinates of single-cell transcriptomics data into an adjacency matrix.

Run the script:
```
python main.py
```
### Results
The final outputs the model's AUC performance, a cell-cell communication matrix, a cell type communication matrix, and results from clustering, UMAP visualization, and PAGA analysis, providing a comprehensive view of cell interactions and relationships.






