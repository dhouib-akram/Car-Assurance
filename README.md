# Car Assurance AI Analysis

## Project Overview
This project applies AI to analyze car insurance claims following accidents. It focuses on image analysis to classify and assess car damage. The approach includes distinguishing AI-generated images from real ones, assessing damage severity, and segmenting car damage using advanced machine learning models.

## Features
- **AI or Not Classification**: Identify whether an image is AI-generated or real.
- **Damage Severity Classification**: Determine the severity of car damage.
- **Car Damage Segmentation**: Use Cascade Mask R-CNN for detailed damage segmentation.

## AI or Not Classification Benchmark Results

| Model         | Accuracy | Mean Average Precision (mAP) |
|---------------|----------|------------------------------|
| VGG16         | 91.32%   |            0.9840            |
| ResNet50      | 66.32%   |            0.9231            |
| Inception V3  | 90.28%   |            0.9902            |
| DenseNet21    | 91.32%   |            0.9886            |

## Damage Severity Classification
| Model         | Accuracy |
|---------------|----------|
| VGG16         | 70.54%   |
| ResNet50      | 0.6473%  |
| Inception V3  | 90.28%   |
| EfficientNetB0|  71.43%  |


## Car Damage Segmentation
The project utilizes Cascade Mask R-CNN for segmenting, detecting, and labeling different types of car damage such as dents, scratches, cracks, etc. This advanced model allows for precise localization and categorization of damage, enhancing the accuracy of insurance claim processing.

## Conclusion
The project demonstrates the potential of AI in automating and improving the accuracy of car insurance claims processing, especially in image classification and damage assessment.

## Additional Resources
For a detailed demonstration of our work, please refer to this video: [Video Link](#)

## Acknowledgments
*(Credits to collaborators, data sources, etc.)*
