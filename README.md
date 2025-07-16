# Custom CNN for Image Classification from Scratch

### Project Overview 🧠

This project presents the end-to-end development of a deep learning model for classifying images into one of **20 distinct categories**. The primary objective was to design, build, and train a **Convolutional Neural Network (CNN)** from the ground up using PyTorch.

This project serves as a practical, hands-on application of fundamental deep learning principles, emphasizing custom architecture design and robust training practices.

---

### File Structure 📂

The repository is organized as follows:

-   **/data**: Contains the training and testing image datasets.
-   **/notebooks**: Includes the Jupyter Notebook with the complete Python code for the model.
-   **/results**: Contains the final prediction output.
-   **README.md**: This report file.

---

### Model Architecture & Training 🛠️

The model is a **custom-coded Convolutional Neural Network (CNN)**. Key architectural features and the training process include:

*(**Your action here:** Briefly describe your CNN. For example: "The architecture consists of three convolutional layers with ReLU activation functions, followed by max-pooling layers. After flattening, two fully-connected layers lead to the final output." You can find this in your `images_clasification_model.ipynb` file.)*

The model was trained from scratch using the following methods:

-   **Framework:** PyTorch
-   **Data Augmentation:** *(Mention any techniques you used, e.g., random flips, rotations)*
-   **Optimizer:** *(e.g., Adam, SGD)*
-   **Loss Function:** *(e.g., Cross-Entropy Loss)*
-   **Best Practices:** The model was trained using early stopping and regularization (Dropout) to prevent overfitting.

For a complete implementation, please see the [**Jupyter Notebook**](./notebooks/images_clasification_model.ipynb).

---

### Results & Evaluation 📊

The model's performance was evaluated based on **classification accuracy** on the unseen test set. The final predictions are available in the `submission.csv` file located in the `results` folder.

*(**Optional - for an even better report:** If you have performance metrics like accuracy or a confusion matrix from your notebook, you can add a screenshot of the plot here.)*

**Example of how to add an image:**
`![Confusion Matrix](path/to/your/confusion_matrix.png)`

---

### How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-project-name.git](https://github.com/your-username/your-project-name.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(**Your action here:** You should create a `requirements.txt` file listing the libraries needed, like `torch`, `torchvision`, `pandas`, `matplotlib`)*
3.  **Open and run the notebook:**
    Navigate to the `notebooks/` directory and open `images_clasification_model.ipynb` in Jupyter Lab or Notebook.
