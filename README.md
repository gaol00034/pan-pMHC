# Structure directed pan-pMHC TCR-pMHC interaction prediction
![workflow](https://github.com/gaol00034/pan-pMHC/blob/main/Figures/Workflow.png)
## Models
![workflow](https://github.com/gaol00034/pan-pMHC/blob/main/Figures/models.png)
## Data description
Except the SEQ-BD and the STRUCT-CS datasets, the other datasets used in our paper could be downloaded from [NetTCR2.2_full_dataset](https://github.com/mnielLab/NetTCR-2.2/blob/main/data/nettcr_2_2_full_dataset.csv) and [IMMREP22*](https://github.com/mnielLab/NetTCR-2.2/blob/main/data/IMMREP/train/all_peptides_redundancy_reduced.csv)
### SEQ-BD
All the positive data are picked from VDJdb[VDJ]() and ITRAP denoised 10x Genomics[TIRAP_10xGenomics](https://github.com/mnielLab/ITRAP_benchmark);
And the negative data are sampled from 10x Genomics[10xGenomics](https://www.10xgenomics.com/datasets);
The ./data/SEQ-BD/sample.csv give the data format of SEQ-BS;
### STRUCT-CS
The structure data are derived from IMGT[IMGT-3DstructDB](https://www.imgt.org/3Dstructure-DB/)
The ./data/STRUCT-CS/PDBids.txt shows all the pdb files used in our work.
## Run

## Contact
If you have any questions, please contact us at [](ltgao34@njust.edu.cn) or [](njyudj@njust.edu.cn)
