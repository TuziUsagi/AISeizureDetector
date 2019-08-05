# EEG seizure detector based on Inception and LSTM

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)


# Model (Inception):
![LSTM model](https://raw.githubusercontent.com/TuziUsagi/AISeizureDetector/master/model.png)
1.	The input data will be up-sampled to create more input points
2.	A fixed filter bank is applied to extract different frequencies components into different channels. This filter bank is generated in scipy and applied as a Conv1D layer and it is not trainable. This filter bank works as a manual feature extraction. 
3.	Four inception blocks are used in parallel. First group of two have kernels with large dilate rate to look for the global features, one of them has maxpool and another has averagepool. Another group of two have kernels with no dilate to look for the local features, again, one of them has maxpool and another has averagepool.
4.	The outputs of four inception blocks are flattened and concatenated, two fully connected layers are used to generate the final output.
5.	Four auxiliary outputs from each inception blocks are also used to prevent gradient vanishing problem. They are ignored during inference.
6.	All lambda layers are used to implement swish activation.
7.	Detailed hyper-parameters are in hparam.py

### Results:

  The network was trained with a training set with ~700,000 samples and validated with a eval set with 190,000 samples and achieved 99.89% validation accuracy and 99.96% training accuracy.

### Directory structure:

| File name | content |
| ------ | ------ |
| hparam.py | defines all the hyper-parameters |
| inception.py | defines the network. There are two versions of network are implemented: *_gpu is intended for GPU training. It can perform double precision filtering for the initial filter bank or mixed precision training, based on the user choices made in hparam.py. *_tpu is intended for tpu training. It requires fixed batch_size for the max performance |
| readDataset.py | defines the input data pipelines. It parses data from tfrecord, perform data augmentation. Parallel processing is enabled by default according to ‘numOfCores’ in hparam.py. Note, the input data is assumed to have zero mean and -1 to 1 range. It also performs the necessary casting if double precision filter bank is needed] |
| trainModel.py | perform the training on CPU or GPU. Please note the validation process on GPU requires dedicated memory space in addition to the training process. If training batch size consumes more than half the memory and the validation process has the same batch size as the training process, you will have the OOM error. To maximize the training memory, validation batch size is reduced by 16x compares with the training batch size. Note, XLA compilation cannot be used with the dynamic batch size setup |
| trainModelTPU.py | perform the training on TPU, take care of the TPU settings. Please note, it must run on google cloud VM. The input dataset must be stored in google cloud storage bucket and the TPU account must have the read permission to the bucket. The batch size must be static. This is the global batch size, the TF server that connected to the TPU will automatically evenly distribute the batch into 8 TPU cores, since each TPU core has an MXU size of 128, the global batch size should be the multiple of 1024.|

### Dependency:

 - Tensorflow 1.14
 - CUDA 10.1
