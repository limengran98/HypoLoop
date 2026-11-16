# Scientific tasks, datasets, and original predictors

| Domain | Task | Dataset / data | Original model / code | Metric | References |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Biology | **BIO01 · IFNg knockout** | [Data](https://singlecell.broadinstitute.org/single_cell/study/SCP1064) | [GEARS](https://github.com/snap-stanford/GEARS) | RNA MSE ↓ | [Dataset](https://doi.org/10.1038/s41588-021-00779-1) · [Model](https://doi.org/10.1038/s41587-023-01905-6) |
| Chemistry | **CHEM01 · ESOL** | [Data](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) | RMSE ↓ | [Dataset](https://doi.org/10.1021/ci034243x) · [Model](https://openreview.net/forum?id=6K2RM6wVqKu) |
| Chemistry | **CHEM02 · Lipophilicity** | [Data](https://doi.org/10.6019/chembl3301361) | [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) | RMSE ↓ | [Dataset](https://doi.org/10.1039/C7SC02664A) · [Model](https://openreview.net/forum?id=6K2RM6wVqKu) |
| Materials | **MAT01 · Exfoliation energy** | [Data](https://github.com/materialsproject/matbench) | [CGCNN](https://github.com/txie-93/cgcnn) | MAE (meV/atom) ↓ | [Dataset](https://doi.org/10.1038/s41524-020-00406-3) · [Model](https://doi.org/10.1103/PhysRevLett.120.145301) |
| Materials | **MAT02 · Phonon peak** | [Data](https://github.com/materialsproject/matbench) | [CGCNN](https://github.com/txie-93/cgcnn) | MAE (cm^-1) ↓ | [Dataset](https://doi.org/10.1038/s41524-020-00406-3) · [Model](https://doi.org/10.1103/PhysRevLett.120.145301) |
| Physics | **PHY01 · Burgers** | [Data](https://github.com/pdebench/PDEBench) | [FNO1d](https://github.com/pdebench/PDEBench/tree/main/pdebench/models/fno) | Relative L2 ↓ | [Dataset](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html) · [Model](https://openreview.net/forum?id=c8P9NQVtmnO) |
| Physics | **PHY02 · Advection** | [Data](https://github.com/pdebench/PDEBench) | [FNO1d](https://github.com/pdebench/PDEBench/tree/main/pdebench/models/fno) | Relative L2 ↓ | [Dataset](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html) · [Model](https://openreview.net/forum?id=c8P9NQVtmnO) |
| Physics | **PHY03 · Reaction-diffusion** | [Data](https://github.com/pdebench/PDEBench) | [FNO1d](https://github.com/pdebench/PDEBench/tree/main/pdebench/models/fno) | Relative L2 ↓ | [Dataset](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html) · [Model](https://openreview.net/forum?id=c8P9NQVtmnO) |
| Energy | **ENER01 · Na-ion battery** | [Data](https://zenodo.org/records/17958489) | [PBT-L](https://github.com/Ruifeng-Tan/PBT) | MAPE (%) ↓ | [Dataset](https://doi.org/10.1145/3711896.3737372) · [Model](https://doi.org/10.1039/D6EE03446B) |
| Energy | **ENER02 · Building load** | [Data](https://github.com/buds-lab/building-data-genome-project-2) | [DLinear](https://github.com/cure-lab/LTSF-Linear) | MAE (kWh) ↓ | [Dataset](https://doi.org/10.1038/s41597-020-00712-x) · [Model](https://doi.org/10.1609/aaai.v37i9.26317) |
| Energy | **ENER03 · Wind power** | [Data](https://doi.org/10.1038/s41597-024-03427-5) | [DLinear](https://github.com/cure-lab/LTSF-Linear) | Turbine-macro MAE ↓ | [Dataset](https://doi.org/10.1038/s41597-024-03427-5) · [Model](https://doi.org/10.1609/aaai.v37i9.26317) |
| Depth | **DEP01 · Make3D** | [Data](https://cs.stanford.edu/people/asaxena/make3d/) | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) | Capture-macro AbsRel ↓ | [Dataset](https://doi.org/10.1109/TPAMI.2008.132) · [Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html) |
| Depth | **DEP02 · DIODE** | [Data](https://diode-dataset.org/) | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth) | Scene-macro AbsRel ↓ | [Dataset](https://arxiv.org/abs/1908.00463) · [Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/26cfdcd8fe6fd75cc53e92963a656c58-Abstract-Conference.html) |
| Seismology | **SEIS01 · STEAD** | [Data](https://github.com/smousavi05/STEAD) | [PhaseNet](https://github.com/seisbench/seisbench) | P-arrival MAE (s) ↓ | [Dataset](https://doi.org/10.1109/ACCESS.2019.2947848) · [Model](https://doi.org/10.1093/gji/ggy423) |
| Mathematics | **MATH01 · Constrained QP** | [Data](https://github.com/locuslab/DC3/tree/main/datasets/simple) | [DC3](https://github.com/locuslab/DC3) | Relative objective gap ↓ | [Dataset](https://openreview.net/forum?id=V1ZHVxJ6dSS) · [Model](https://openreview.net/forum?id=V1ZHVxJ6dSS) |
| Mathematics | **MATH02 · Nonconvex optimization** | [Data](https://github.com/locuslab/DC3/tree/main/datasets/nonconvex) | [DC3](https://github.com/locuslab/DC3) | Relative objective gap ↓ | [Dataset](https://openreview.net/forum?id=V1ZHVxJ6dSS) · [Model](https://openreview.net/forum?id=V1ZHVxJ6dSS) |
| Time series | **TS01 · Beijing PM2.5** | [Data](https://doi.org/10.24432/C5RK5G) | [NLinear](https://github.com/cure-lab/LTSF-Linear) | MAE ↓ | [Dataset](https://doi.org/10.24432/C5RK5G) · [Model](https://doi.org/10.1609/aaai.v37i9.26317) |
| Industrial inspection | **ENG01 · MVTec AD** | [Data](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | [AnomalyCLIP + R](https://github.com/zqhang/AnomalyCLIP) | Pixel AP ↑ | [Dataset](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html) · [Model](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html) |
| Industrial inspection | **ENG02 · MVTec LOCO** | [Data](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | [AnomalyCLIP + R](https://github.com/zqhang/AnomalyCLIP) | Pixel AP ↑ | [Dataset](https://doi.org/10.1007/s11263-022-01578-9) · [Model](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html) |
| Industrial inspection | **ENG03 · VisA** | [Data](https://registry.opendata.aws/visa/) | [AnomalyCLIP + R](https://github.com/zqhang/AnomalyCLIP) | Pixel AP ↑ | [Dataset](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2149_ECCV_2022_paper.php) · [Model](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html) |
| Industrial inspection | **ENG04 · MPDD** | [Data](https://github.com/stepanje/MPDD) | [AnomalyCLIP + R](https://github.com/zqhang/AnomalyCLIP) | Pixel AP ↑ | [Dataset](https://doi.org/10.1109/ICUMT54235.2021.9631567) · [Model](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d7b50b8ac2c781a12f26155f48310d8d-Abstract-Conference.html) |
| Remote sensing | **RS01 · SpaceNet7** | [Data](https://spacenet.ai/sn7-challenge/) | [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba) | Change F1 ↑ | [Dataset](https://openaccess.thecvf.com/content/CVPR2021/html/Van_Etten_The_Multi-Temporal_Urban_Development_SpaceNet_Dataset_CVPR_2021_paper.html) · [Model](https://doi.org/10.1109/TGRS.2024.3417253) |

## Predictor configuration

- **GEARS:** gene-perturbation prediction on the IFNg RNA condition; whole target-gene groups.
- **Uni-Mol:** property-specific molecular regression, with scaffold-grouped feedback validation.
- **CGCNN:** exfoliation energy and phonon-peak property models, grouped by chemical formula.
- **FNO1d:** equation-specific PDEBench predictors; Burgers, advection and reaction-diffusion.
- **PBT-L:** the MIX_large battery-life predictor with the BatteryLife Na-ion collection.
- **DLinear / NLinear:** 96-step input and 24-step forecast; DLinear moving-average window 25.
- **Depth Anything V2:** metric ViT-S VKITTI checkpoint, with original sigmoid and metric scale.
- **PhaseNet:** the SeisBench SCEDC checkpoint and native three-class prediction path.
- **DC3:** native completion and correction for QP and nonconvex constrained optimization.
- **AnomalyCLIP + R:** the VisA checkpoint with a shared zero-initialized 769-parameter residual readout.
- **ChangeMamba:** the LEVIR-CD+ checkpoint and native building-change segmentation path.

The paper uses task-specific original checkpoints and source-partition adaptations. Runtime feedback, validation groups, native preprocessing, and the corresponding task metric are configured through the [native task interface](NATIVE_TASKS.md).

## Metric conventions

The released tables contain native errors and AP/F1 fractions. Error improvements are relative percentages; AP/F1 improvements are percentage-point differences. Industrial AP is computed on full-resolution masks and aggregated by category. Building load uses building-macro MAE in kWh. Depth uses the registered capture/scene aggregation and valid support. QP uses the OSQP objective reference; nonconvex optimization uses its IPOPT local reference.

Please cite the dataset and predictor publications linked in each row. BibTeX entries are in [references.bib](../references.bib).
