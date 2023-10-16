# Introduction to key modules

- data structure

MMSegmentation 1.0 introduces the SegDataSample data structure, which encapsulates the data in semantic segmentation and is used for data transfer between various functional modules. There are three fields in SegDataSample: gt_sem_seg, pred_sem_seg and seg_logits. The first two are segmentation masks corresponding to labels and model predictions, and seg logits are the unnormalized output of the last layer of the model.
