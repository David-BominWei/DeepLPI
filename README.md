# DeepLPI
A deep learning based drug target interaction prediction model

## file description

+ `/scipts`: File inside this folder is the jupyter notebook (`.py`) copy of the `.ipynb` file. 

    + `/scripts/runtrainingDeepLPI6165.sh`: Run the training of the model. **NOTICE: It will remove the original model saved in `/output`**

+ `env.list`: environment file, the current environment for developing the model

+ `step0_preprocess.ipynb`: the data preprocessing file, run this to preprocess the original data.

+ `step1_molembedding.ipynb`: use to embedding molecular

    + requirement: Mol2Vec

+ `step2_seqembedding.ipynb`: use to embedding sequence

    + requirement: ProSE

+ `DeepLPI_6165_Bin.py`: The model training file, save the configuration and output in `/output` folder after training for 1000 epochs

    + requirement: processed data

+  `DeepLPI_6165_Bin_Evaluation.ipynb`: Evaluate the performance of the model, the file will also draw the necessary figures and save in `/output`

## model description

## updates

+ 2022/10/24
    + building revision of `DeepLPI_6165_Reg.ipynb` and `DeepLPI_6165_Reg-Daviscp.ipynb`

+ 2022/10/21
    + rename `DeepLPI_6165kdCl_Evaluation.ipynb` -> `DeepLPI_6165_Bin_Evaluation.ipynb`
    + rename `DeepLPI_6165_Kd_Classification.ipynb` -> `DeepLPI_6165_Bin.ipynb`
    + evaluate `DeepLPI_6165_Bin` model

+ 2022/10/20
    + revised LSTM module in branch `6165_LSTM_Revise`

+ 2022/10/19
    + create `DeepLPI_6165kdCl_Evaluation.ipynb` to evaluate the model

+ 2022/10/18
    + train the original model `DeepLPI_6165_Kd_Classification.ipynb`
    + update the `README.md`, start to write updates