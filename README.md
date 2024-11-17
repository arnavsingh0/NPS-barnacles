# National Park Service - Barnacles
## DALI Challenge
### Author: Arnav Singh

## Overview

This repository contains the code and documentation for a prototype system designed to help Natinoal Park Service scientists count barnacles in images more efficiently. Effectively, the goal is to automate the process of counting barnacles within a fixed-size frame in images, reducing the time and effort required for laborious manual counting, so that scientists can use their time for the greater good. 

The project explores various approaches, including traditional computer vision techniques, basic machine learning models, and from a paper I read, advanced deep learning models like Vision Transformers (ViTs). The repository includes a Jupyter Notebook that documents the entire process, from data preprocessing to the model evaluation.

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

1. **Data Preprocessing**: The images and masks are loaded, normalized, and resized to a consistent size for processing.
2. **Exploratory Data Analysis (EDA)**: Visualizations of the images and masks are created to understand the data better.
3. **Basic Model Exploration**: A simple Convolutional Neural Network (CNN) is implemented to establish a baseline for barnacle detection.
4. **Advanced Model (Vision Transformer)**: A Vision Transformer (ViT) is implemented to explore its potential for higher accuracy.
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

