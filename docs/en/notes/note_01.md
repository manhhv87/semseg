# Introduction to key modules

- data structure

MMSegmentation 1.0 introduces the SegDataSample data structure, which encapsulates the data in semantic segmentation and is used for data transfer between various functional modules. There are three fields in SegDataSample: gt_sem_seg, pred_sem_seg and seg_logits. The first two are segmentation masks corresponding to labels and model predictions, and seg logits are the unnormalized output of the last layer of the model.

- Dataset and data change operations

MMSegmentation supports a total of 14 data sets. In addition to common academic data sets such as ADE20k and CityScapes, there are also application-oriented data sets such as medical and optical aerial remote sensing.

MMSegmentation 1.0 newly defines BaseSegDataset, standardizes the functions and interfaces of semantic segmentation data sets, and is a subclass of BaseDataset in MMEngine. The main function of the data set is to load data information. There are two types of data information. One is the meta-information of the data set, including category information and palette information, which is the color corresponding to the category when rendering; the other is data information, which is saved. The image path and corresponding label path in the specific data set are specified.

The dataset module contains a data transformation module, which supports many data transformations. During training or testing, a series of data transformations can be combined into a list, called a data pipeline, and passed to the pipeline parameters of the dataset, each module in the pipeline. The output is the input to the next module.
The figure below shows the data transformation pipeline during training of a typical semantic segmentation model. For each pair of samples, after a data transformation operation, new fields (marked in green) or existing fields (marked in orange) will be added to the output dictionary.

