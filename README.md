  We propose an innovative semi-supervised medical image segmentation approach that distinguishes and fully utilizes all pseudo-labels while effectively capturing and leveraging the differences between decoders for layer-wise gradient updates. Specifically, we introduce a comparison network to differentiate pseudo-labels and employ diverse strategies to learn pseudo-labels of varying reliability. Next, we enhance the encoder's ability to learn from the differences revealed by different decoder perspectives on the same unlabeled data. Finally, we propose distinct rules for the classification layers in the decoder when learning pseudo-labels and integrate a parallel Mamba module with the CNN module in the encoder. By comprehensively utilizing various aspects of pseudo-label information, our method maximizes the potential of pseudo-labels. We validate the effectiveness of our approach through extensive experiments on the LA, Pancreas-NIH, and Brats-2019 datasets, demonstrating state-of-the-art performance.
 ## Dataset Download Instructions
This project uses multiple medical image datasets. You can obtain them from the following sources:

### 1. LA Dataset (Left Atrium)

	Available on GitHub: https://github.com/yulequan/UA-MT/tree/master/data

### 2. BraTS 2019 (Brain Tumor Segmentation)
	Available via Baidu NetDisk: data.zip
	Link: https://pan.baidu.com/s/10S_MzY-DJCivrjjF-BfoRw
	Access Code: sq3d

### 3. Pancreas CT Dataset
	Available via Baidu NetDisk: Pancreas-CT.zip
	Link: https://pan.baidu.com/s/1q0C06RFd-g1vr7S08RmSsA
	Access Code: 39pu

