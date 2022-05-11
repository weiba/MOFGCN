
# ReadMe

These data are used in the MOFGCN algorithm.



## GDSC



- cell_drug.csv records the log IC50 association matrix of cell line-drug. 
- cell_drug_binary.csv records the binary cell line-drug association matrix. 
- cell_cna/ records the CNA features of the cell line. Due to a large amount of raw data, we use the code in the file utils.py to split.
- cell_gene/ records cell line gene expression features. Due to a large amount of raw data, we use the code in the file utils.py to split.
- cell_mutation/ records somatic mutation features of cell lines. Due to a large amount of raw data, we use the code in the file utils.py to split.
- drug_feature.csv records the fingerprint features of drugs. 
- null_mask.csv records the null values in the cell line-drug association matrix. 
- threshold.csv records the drug sensitivity threshold.



## CCLE

- cell_drug.csv records the log IC50 association matrix of cell line-drug. 
- cell_drug_binary.csv records the binary cell line-drug association matrix. 
- cell_cna/ records the CNA features of the cell line. Due to a large amount of raw data, we use the code in the file utils.py to split.
- drug_feature.csv records the fingerprint features of drugs. 
- cell_gene/ records cell line gene expression features. Due to a large amount of raw data, we use the code in the file utils.py to split.
- cell_mutation.csv records somatic mutation features of cell lines.



## utils.py

### Split file

```python
gdsc_dir = './GDSC/'
ccle_dir = './CCLE/'
for name in os.listdir(gdsc_dir):
  path = gdsc_dir + name
  print(path)
  zip_trans = CsvLimitSizeZip(path=path)
  zip_trans.split_file()
  for name in os.listdir(ccle_dir):
    path = ccle_dir + name
    print(path)
    zip_trans = CsvLimitSizeZip(path=path)
    zip_trans.split_file()
```



### Merge file

```python
gdsc_dir = './GDSC/'
path = gdsc_dir + "cell_gene.csv"
zip_trans = CsvLimitSizeZip(path=path)
zip_trans.merge_file(path.rstrip('.csv') + '/')
```

