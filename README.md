# National Park Service - Barnacles
## DALI Challenge
### Author: Arnav Singh

## Overview

This repository contains the code and documentation for a prototype system designed to help Natinoal Park Service scientists count barnacles in images more efficiently. Effectively, the goal is to automate the process of counting barnacles within a fixed-size frame in images, reducing the time and effort required for laborious manual counting, so that scientists can use their time for the greater good. 

The project explores various approaches, including traditional computer vision techniques, basic machine learning models, and advanced deep learning models like Vision Transformers (ViTs) from a paper I read from researchers at Google Brain (https://arxiv.org/abs/2010.11929). The repository includes a Jupyter Notebook that documents the entire process, from data preprocessing to the model evaluation.

---

## Repository Structure

```
barnacle-counting-challenge/
├── README.md                   # This file
├── barnacle.ipynb              # Main Jupyter Notebook
├── requirements.txt            # List of dependencies
├── Barnacles/            # Directory for given data
│   ├── img1.png
│   ├── img2.png
│   ├── mask1.png
│   ├── mask2.png
│   └── unseen_img1.png
└── utils/                      
    └── preprocessing.py
```

---

## First thoughts for an Approach

First we can tackle this challenge by looking at each segment of a typical ML pipeline:

1. **Data Loading & Preprocessing**: The images and masks need to be loaded, normalized, and resized to a consistent size for processing. Additionally, the dataset given does not include a lot of images, rather we have to somehow create our own to train the models to count barnacles. For this task, we can crop the framed image into even smaller segments, which should be easier for the model to identify the contours. Additionally, a trick is to rotate the images to basically double/triple/quadruple our dataset for training. Overall, from the two images we have `img1.png` and `img2.png`, we should be able to have >200 samples to train our model on.

2. **Exploratory Data Analysis (EDA)**: Visualizations of the images and masks are created to understand the data better.

3. **Basic Model Exploration**: A simple Convolutional Neural Network (CNN) is implemented to establish a baseline for barnacle detection.

4. **SOTA Model (Vision Transformer)**: A Vision Transformer (ViT) is implemented to explore its potential for higher accuracy.

5. **Results and Analysis**: The performance of both models is evaluated and compared.


6. **Conclusions and Next Steps**: Insights from the prototype are summarized, and potential improvements are discussed.

---

## How to Run the Code

### Prerequisites

1. **Install Dependencies**: Ensure you have the necessary libraries installed. You can install them using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Data**: Place the provided images and masks in the `Barnacles/` directory. The required files are:
   - `img1.png`, `img2.png` (example images)
   - `mask1.png`, `mask2.png` (corresponding masks)
   - `unseen_img1.png` (unlabeled test image)

3. **Run the Jupyter Notebook**:
   - Open the `barnacle.ipynb` notebook in Jupyter or any compatible environment (e.g., VSCode, Google Colab).
   - Run the cells sequentially to execute the code and see the results.

---

## Dependencies

The project uses the following Python libraries:
- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `transformers` (for Vision Transformer)

You can install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Results

- **Basic CNN Model**: Achieved a reasonable accuracy on the task of barnacle detection, serving as a good baseline.
- **Vision Transformer (ViT)**: Showed potential for higher accuracy, especially with more complex data. However, it requires more computational resources and fine-tuning.

---

## Conclusions

The prototype demonstrates that automating barnacle counting is feasible using machine learning and deep learning techniques. While the basic CNN model provides a solid starting point, the Vision Transformer offers a promising direction for future work. Key takeaways include:
- Data preprocessing and augmentation are critical for improving model performance.
- Advanced models like ViTs can achieve higher accuracy but require more data and computational resources.
- Integrating human feedback into the pipeline could further enhance the system's reliability.

---

## Learning and Next Steps

This project was an excellent opportunity to explore various machine learning and deep learning techniques for image processing. Key learnings include:
- Understanding the challenges of working with limited annotated data.
- Gaining hands-on experience with Vision Transformers, inspired by recent research papers.
- Learning how to evaluate and compare different models effectively.

**Next Steps**:
1. Collect more annotated data to improve model training.
2. Fine-tune the Vision Transformer for better performance.
3. Develop an interactive interface (e.g., using Streamlit) to allow scientists to correct model predictions and provide feedback.

---

## Submission Notes

This repository is part of my application for the DALI Lab. The code and documentation demonstrate my ability to approach an open-ended data problem, explore different solutions, and implement a functional prototype. I hope this work showcases my immense enthusiasm for data science and machine learning, in a way that you all found wonderful!

---

## Contact

If you have any questions or feedback, feel free to reach out:
- **Name**: Arnav Singh
- **Email**: arnav.singh.26@dartmouth.edu
- **GitHub**: arnavsingh0