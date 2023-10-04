## Environment:

* GPU: NVIDIA Tesla V100 SXM2
* Python: python3.9

## Installation

```bash
conda create -n biorex python=3.9
conda activate biorex
pip install -r requirements.txt
```

## Train and evaluate BioREx

### Step 1: Download the datasets for BioREx

You can download [our converted datasets](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/datasets.zip), and unzip it to 

```
datasets/
```

If you want to convert the datasets by yourself, you can use the below script to convert original datasets into our input format.
```
bash scripts/build_biorex_datasets.sh
```

### Step 2: Download the pre-trained model

Please download the model [here](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)

Then put them into 

```
microsoft/
```

### Step 3: Train and evaluate

```
bash scripts/run_biorex_exp.sh
```

## BioREx pre-trained models

You can download the BioREx pre-trained models:

* [BioREx BioLinkBERT model (Preferred)](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model_biolinkbert.zip) is utilized in the beta version of [PubTator3](https://www.ncbi.nlm.nih.gov/research/pubtator3/).
* [BioREx PubMedBERT model (Original)](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model.zip) 

## Predicting New Data:

If you only wish to use our tool for predicting new data without the need for training, please follow the steps outlined below:

Download the BioREx pre-trained model [BioREx model](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model.zip) file and place it in the "BioREx/" directory.

Open the "scripts/run_test_pred.sh" file and modify the values of the variables "in_pubtator_file" and "out_pubtator_file" to match your input PubTator file (with annotations) and the desired output PubTator file (where predicted relations will be stored).

Execute the following script to initiate the prediction process:

```
bash scripts/run_test_pred.sh <CUDA_VISIBLE_DEVICES>
```

Please replace the above <CUDA_VISIBLE_DEVICES> with your GPUs' IDs. Eg: '0,1' for GPU devices 0 and 1.
For example

```
bash scripts/run_test_pred.sh 0,1
```

## Citing BioREx

* Lai P. T., Wei C. H., Luo L., Chen Q. and Lu Z. BioREx: Improving Biomedical Relation Extraction by Leveraging Heterogeneous Datasets. Journal of Biomedical Informatics. 2023.
```
@article{lai2023biorex,
  author  = {Lai, Po-Ting and Wei, Chih-Hsuan and Luo, Ling and Chen, Qingyu and Lu, Zhiyong},
  title   = {BioREx: Improving Biomedical Relation Extraction by Leveraging Heterogeneous Datasets},
  journal = {Journal of Biomedical Informatics},
  volume  = {146},
  pages   = {104487},
  year    = {2023},
  issn    = {1532-0464},
  doi     = {https://doi.org/10.1016/j.jbi.2023.104487},
  url     = {https://www.sciencedirect.com/science/article/pii/S1532046423002083},
}
```

## Acknowledgments

This research was supported by the NIH Intramural Research Program, National Library of Medicine. It was also supported by the National Library of Medicine of the National Institutes of Health under award number 1K99LM014024 (Q. Chen) and the Fundamental Research Funds for the Central Universities [DUT23RC(3)014 to L.L.].

## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.
