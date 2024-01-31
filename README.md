## Explanation cifar10 scripts:

- Cross_validate_gpfq.py: Cross validate C_alpha hyperparameter of GPFQ scheme
- plot_quant_error.py: Plot the quantization error of the GPFQ, OAQ and MSQ quantizations
- preprocess.py: Preprocess algorithm of the OAQ scheme
-quant_error_comp.py: Generate data for the quantization of the NN using the GPFQ, OAQ and MSQ scheme
- quantized_net.py, quantized_net_gpfq.py, quantized_net_msq.py: Classes to handle neural networks and the quantized version thereof
- train_cifar10.py: Script to train a NN on the CIFAR10 dataset.

## Explanation cifar10 data:

- gpfq_cross_validation folder: Contains two csv files with tables of the relative quantization error corresponding to different bitsizes and alphabet scalars
- thesiss folder: Contains the tables of the relative quantization error of layer 1 and layer 2 and the plot thereof
- model.keras: Pre-trained Keras model

## Explanation mnist scripts:

- comp_plot.py: Plot the quantization error of the GPFQ, OAQ and MSQ quantizations
- Cross_validate_gpfq.py: Cross validate C_alpha hyperparameter of GPFQ scheme
- deeper_layer_comp.py: Generate data for the preprocess input (analog or quantized) of deeper layers
- preprocess.py: Preprocess algorithm of the OAQ scheme
- quant_error_comp.py: Generate data for the quantization of the NN using the GPFQ, OAQ and MSQ scheme
- quantized_net.py, quantized_net_gpfq.py, quantized_net_msq.py: Classes to handle neural networks and the quantized version thereof
- train_mnist: Script to train a NN on the MNIST dataset

## Explanation mnist data:

- gpfq cross validate folder: Contains csv files of the relative quantization error corresponding to different alphabet scalars
- plot and data 5 bit: Contains the data and the plot for the quantization error in the different layers using different quantization schemes for 5 bit quantization
- table_log_23_bits: Contains the data for the quantization error in the different layers using different quantization schemes for log_2(3) quantization
- model.keras: Pre-trained Keras model

## Explanation normal distributed data:

- normal_distr_test: Contains the folders with the tables of the quantization errors for the GPFQ, OAQ and MSQ schemes

## Explanation runtime tests

- runtime_tests: Contains the plot and used matrices for the comparsion of the runtime