# Structure directed pan-pMHC TCR-pMHC interaction prediction
![workflow](https://github.com/gaol00034/pan-pMHC/blob/main/Figures/Workflow.png)
## Models
![workflow](https://github.com/gaol00034/pan-pMHC/blob/main/Figures/models.png)
***
## Data description
Except the SEQ-BD and the STRUCT-CS datasets, the other datasets used in our paper could be downloaded from [NetTCR2.2_full_dataset](https://github.com/mnielLab/NetTCR-2.2/blob/main/data/nettcr_2_2_full_dataset.csv) and [IMMREP22*](https://github.com/mnielLab/NetTCR-2.2/blob/main/data/IMMREP/train/all_peptides_redundancy_reduced.csv)
### SEQ-BD
All the positive data are picked from [VDJdb](https://vdjdb.cdr3.net/overview) and ITRAP denoised [TIRAP_10xGenomics](https://github.com/mnielLab/ITRAP_benchmark);  
And the negative data are sampled from [10xGenomics](https://www.10xgenomics.com/datasets);  
The [./data/SEQ-BD/sample.csv](https://github.com/gaol00034/pan-pMHC/data/SEQ-BD/sample.csv) give the data format of SEQ-BD.
### STRUCT-CS
The structure data are derived from [IMGT-3DstructDB](https://www.imgt.org/3Dstructure-DB/);  
The [./data/STRUCT-CS/PDBids.txt](https://github.com/gaol00034/pan-pMHC/data/STRUCT-CS/PDBids.txt) shows all the pdb files used in our work.
***
## Run
For training the TCR-pMHC binding specificity, please run the following command
```
python ./Scripts/train.py --train_data_path ./data/SEQ-BD/train.py --epoch 100 --params <param_path> --model_path <model_save_path/model.pkl>
```
The input data format could be refered from in ./data/SEQ-BD/sample.csv
For training the TCR binding site prediction model, please run the following command
```
python ./Scripts/site_train.py --train_data_path ./data/STRUCT-CS/train.py --epoch 100 --params <param_path> --model_path <model_save_path/model.pkl>
```
For predicting the TCR-pMHC bindingn specificity score, please run the following command
```
python ./Scripts/predict.py --model_path <model_save_path/model.pkl> --input ./data/SEQ-BD/predict.py --output <output_path/prediction.csv>
```
The input data format is the same as the format in ./data/SEQ-BD/sample.csv but without the label.
***
## Contact
If you have any questions, please contact us at [ltgao34@njust.edu.cn](ltgao34@njust.edu.cn) or [njyudj@njust.edu.cn](njyudj@njust.edu.cn)
