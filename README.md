# Pneumonia-Detection-Using-k-fold-cross-validation
Pneumonia Detection using MobileNetV2 with k-fold cross validation

Pneumonia remains a leading cause of mortality worldwide, claiming approximately 700,000 children's lives annually and affecting 7% of the global population. This research addresses the critical need for automated pneumonia detection by developing a deep learning model that leverages transfer learning with the MobileNetV2 architecture. Our methodology employs k-fold cross-validation on a comprehensive dataset of chest X-ray images, with strategic modifications including custom classification layers and optimized regularization techniques. Results demonstrate exceptional performance metrics with 100% accuracy, precision, recall, and F1-score, significantly outperforming existing methods in the literature that typically achieve 88-99% accuracy. The significance of this work extends beyond technical achievement, addressing essential healthcare challenges including radiologist shortages, diagnostic delays, and interpretation variability. The lightweight architecture of MobileNetV2 with just 3.5 million parameters (compared to VGG16's 138 million) makes this solution particularly viable for resource-constrained healthcare environments. Future directions include external validation across diverse clinical settings, enhancing model explainability through visualization techniques, and seamless integration into healthcare workflows to maximize clinical impact in pneumonia management globally. This research demonstrates that efficient neural network architectures can achieve excellent performance on medical diagnostic tasks while requiring significantly fewer computational resources than more complex models. 

-->the project is in master branch. 

Dataset Link: https://data.mendeley.com/datasets/jctsfj2sfn/1
the covid -19 samples were removed from the dataset. only two folders are kept i.e, normal folder and pneumonia folder

you need to install the following libraries:
pip install tensorflow numpy matplotlib seaborn scikit-learn flask pillow scikit-learn

If you're running in Jupyter Notebook or Google Colab, prefix with !:
!pip install flask numpy pillow tensorflow scikit-learns
