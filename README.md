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

## BioREx pre-trained model

You can download the BioREx pre-trained model [BioREx model](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/pretrained_model.zip).

You can run BioREx without training if you want to test on your own dataset only. Then replace the [BioRED's pubtator format](datasets/ncbi_relation/Test.PubTator) with your own dataset.

In scripts/run_biorex_exp.sh, you have to replace the variable 'pre_train_model' and remove the parameter 'do_train' .

## Acknowledgments

This research was supported by the Intramural Research Program of the National Library of Medicine (NLM), National Institutes of Health.

## Disclaimer
This tool shows the results of research conducted in the Computational Biology Branch, NCBI. The information produced on this website is not intended for direct diagnostic use or medical decision-making without review and oversight by a clinical professional. Individuals should not change their health behavior solely on the basis of information produced on this website. NIH does not independently verify the validity or utility of the information produced by this tool. If you have questions about the information produced on this website, please see a health care professional. More information about NCBI's disclaimer policy is available.