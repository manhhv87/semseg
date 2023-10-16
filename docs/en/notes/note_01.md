# Introduction to key modules

- **Data structure**

MMSegmentation 1.0 introduces the SegDataSample data structure, which encapsulates the data in semantic segmentation and is used for data transfer between various functional modules. There are three fields in SegDataSample: gt_sem_seg, pred_sem_seg and seg_logits. The first two are segmentation masks corresponding to labels and model predictions, and seg logits are the unnormalized output of the last layer of the model.

- **Dataset and data change operations**

MMSegmentation supports a total of 14 data sets. In addition to common academic data sets such as ADE20k and CityScapes, there are also application-oriented data sets such as medical and optical aerial remote sensing.

MMSegmentation 1.0 newly defines BaseSegDataset, standardizes the functions and interfaces of semantic segmentation data sets, and is a subclass of BaseDataset in MMEngine. The main function of the data set is to load data information. There are two types of data information. One is the meta-information of the data set, including category information and palette information, which is the color corresponding to the category when rendering; the other is data information, which is saved. The image path and corresponding label path in the specific data set are specified.

The dataset module contains a data transformation module, which supports many data transformations. During training or testing, a series of data transformations can be combined into a list, called a data pipeline, and passed to the pipeline parameters of the dataset, each module in the pipeline. The output is the input to the next module.
The figure below shows the data transformation pipeline during training of a typical semantic segmentation model. For each pair of samples, after a data transformation operation, new fields (marked in green) or existing fields (marked in orange) will be added to the output dictionary.

![data_trans](figures/data_transformation_pipeline.awebp)

- **Model**

In MMSegmentation, a semantic segmentation algorithm model is called segmentor, and inherits OpenMMLab's consistent modular design, segmentor is divided into 6 modules, namely:

> - `data_preprocessor` is responsible for transporting the data output from the data transformation pipeline to the designated device, and performing operations such as normalization, padding, and batching on it. The advantage of this is that during the data change stage, the data format is uint8. After the data is transferred to the GPU, it is then converted to FP32 for normalization, which reduces the computing pressure on the CPU.

> - `Backbone` extracts feature maps from input images. Common models include ResNet, Swin transformer, etc.

> - `Neck`, connects backbone and decode_head, further processes the feature map output from backbone and then inputs it to the decoding head. Common networks include Feature Pyramid Network FPN.

> - `decode_head` is responsible for predicting the final segmentation result from the input feature map.

> - `auxiliary_head (optional)` is responsible for predicting segmentation results from the input feature map, but the results only participate in loss calculation during the training process and do not participate in reasoning. During reasoning, the output results are only predicted from the decoding head.

> - `Loss` is responsible for the loss calculation of the neural network output results and true values, and is used for model gradient calculation during backpropagation.

The model structure of segmentor is divided into encoder_decoder and cascade_encoder_decoder according to whether it is connected by multiple decode_head sets. The difference between the two is: there are multiple decode_heads in cascade_encoder_decoder, and starting from the second decoding head, the input of each decoding head is the output of the previous decoding head, and its function is to further refine the output results.

<p align="center">
  <img src="figures/encoder_decoder_data_stream.awebp" />
</p>

encoder_decoder data stream

![cascade_encoder_decoder](figures/cascade_encoder_decoder_data_flow.awebp)

cascade_encoder_decoder data flow
