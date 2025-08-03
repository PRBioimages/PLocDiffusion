# PLocDiffusion
code for “Ying-Yi Wang, Yu Li, Yi-Lin Li, Ying He, Ying-Ying Xu, PLocDiffusion: Diffusion-Based Generation of Cell Images with Quantitative Fluorescence for Precise and Interpretable Protein Subcellular Localization”
Contact: Ying-Ying Xu, yyxu@smu.edu.cn 2025/08/01
# 1.Data preprocess
The data source comes from the subcellular section in the Human Protein Atlas (HPA, https://proteinatlas.org) .Run the codes to get the single-cell images.
# 2.PLocDiffusion
Run .\train.py to train a generative model. And the model will be applied to the unmixing model.
# 3.Unmixing model
The model is based on Bestfitting, can be obtained by 'https://github.com/CellProfiling/HPA-competition-solutions/tree/master/bestfitting'. Run the .\run\generate_quantitative_image.py to get the quantitative images dataset for the unmixing model. After adding quantitative data to the original dataset， run the .\run\train_diffusion to get a quantitative prediction model.
