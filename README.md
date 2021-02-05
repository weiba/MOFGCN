README
===============================
MOFGCN:Predicting Drug Response Based on Multi-omics Fusion and Graph Convolution
This document mainly introduces the python code of MOFGCN algorithm.

# Requirements
- pytorch==1.6.0
- tensorflow==2.3.1
- numpy==1.17.3+mkl
- scipy==1.4.1
- pandas==0.25.2
- scikit-learn=0.21.3
- pubchempy==1.0.4
- seaborn==0.10.0
- hickle==4.0.1
- keras==2.4.3

# Instructions
This project contains all the codes for MOFGCN and 5 comparison algorithms to experiment on the CCLE and GDSC databases, respectively.We only introduce the algorithm proposed in our paper, MOFGCN, and the introduction of other algorithms can be found in the corresponding paper.

# Model composition and meaning
MOFGCN is composed of common modules and experimental modules.
## Common module
- model.py defines the complete MOFGCN model.
- optimizer.py defines the optimizer of the model.
- myutils.py defines the tool functions needed by the entire algorithm during its operation.

## Experimental module
- Entire_Drug_Cell performs the random clearing cross-validation experiment.
	- entire_main.py performs a complete MOFGCN algorithm, which combines gene expression, copy number variation and somatic mutation as cell similarity.
	- entire_gene_main.py performs an experiment that uses only gene expression to calculate cell line similarity.
	- sampler.py defines the sampler for random zeroing experiments.
	- result_data and statistic_result folders save the output results and statistical results of the algorithm respectively.

- New_Drug_Cell performs single row and single column clearing experiments.
	- main.py performs single row and single column clearing experiments.
	- MOFGCN_New_target.py integrates the MOFGCN algorithm.
	- sampler.py defines the sampler for single row and single column clearing experiments.
	- result_data and statistic_result folders save the output results and statistical results of the algorithm respectively.

- Single_Drug_Cell performs a single drug response prediction experiment.
	- main.py performs a single drug response prediction experiment.
	- MOFGCN_Single_target.py integrates the MOFGCN algorithm.
	- sampler.py defines the sampler for single drug response prediction experiment.
	- pan_reslt_data and statistic_result folders save the output results and statistical results of the algorithm respectively.

- Target_Drug performs targeted drug experiments.
	- target_main.py performs targeted drug experiments.
	- sampler.py defines the sampler for targeted drug experiments.
	- result_data and statistic_result folders save the output results and statistical results of the algorithm respectively.

All *main*.py files can complete a single experiment. Because of the randomness of dividing test data and training data, we recorded the true value of the test data during the algorithm performance. Therefore, the output of the main file includes the true and predicted values of the test data that have been cross-validated many times. In the subsequent statistical analysis, we analyze the output of the main file. The myutils.py file contains all the tools needed for the performance and analysis of the entire experiment, such as the calculation of AUC, ACC, F1 score, and MCC. All functions are developed using PyTorch and support CUDA.

Both the CCLE and GDSC folders contain the processed_data file, which contains the input data required by the MOFGCN algorithm and the comparison algorithm.
-GDSC/processed_data/
	- cell_drg_common.csv records the log IC50 association matrix of cell line-drug.
	- cell_drug_common_binary.csv records the binary cell line-drug association matrix.
	- cell_gene_cna.csv records the CNA features of the cell line.
	- cell_gene_feature.csv records cell line gene expression features.
	- cell_gene_mutation.csv records somatic mutation features of cell lines.
	- cell_id_tag.csv records the COSMIC ID of the cell line.
	- drug_cid.csv records the PubChem IDs of all drugs screened in GDSC.
	- drug_feature.csv records the fingerprint features of drugs.
	- null_mask.csv records the null values in the cell line-drug association matrix.
	- threshold.csv records the drug sensitivity threshold.


-CCLE/processed_data/
	- cell_drug.csv	records the log IC50 association matrix of cell line-drug.
	- cell_drug_binary.csv records the binary cell line-drug association matrix.
	- cna_feature.csv records the CNA features of the cell line.
	- drug_feature.csv records the fingerprint features of drugs.
	- drug_name_cid.csv records the drug name and PubChem ID.
	- gene_feature.csv records cell line gene expression features.
	- mutation_featre.csv records somatic mutation features of cell lines.


# Contact
If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me (weipeng1980@gmail.com).
