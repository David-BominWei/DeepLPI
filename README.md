# DeepLPI
A deep learning based drug target interaction prediction model

## file description

+ `Model/`: folder of all models
    + `archive/`: archived models
    + `notebook/`: notebook version of the models, used for testing.
    + model files:
        ```sh
        $modelname + $mainversion + $subversion + "-" + $database + "-" + $mode
        ```

+ `Evaluation/`: evaluation of the model

+ `Preprocess/`: preprocessing of the model, plz follow the step.

+ `output/`: data output

+ `env.list`: python environment requirement

## model description

+ `v9b0`: original version
+ `v9b1`: LSTM improved version

## updates

+ 2022/10/24
    + `6165_reg_Revise` test fail => del
        + move back to main without merge the branch
    + reorganized the file(see file description)

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