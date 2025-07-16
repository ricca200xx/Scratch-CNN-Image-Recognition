# Custom CNN for Image Classification from Scratch

### Project Overview 🧠

This project presents the end-to-end development of a deep learning model for classifying images into one of **20 distinct categories**. The primary objective was to design, build, and train a **Convolutional Neural Network (CNN)** from the ground up using PyTorch.

This project serves as a practical, hands-on application of fundamental deep learning principles, emphasizing custom architecture design and robust training practices.

---

### File Structure 📂

The repository is organized as follows:

-   **/notebooks**: Includes the Jupyter Notebook with the complete Python code for the model.
-   **/results**: Contains the final prediction output (`submission.csv`).
-   **README.md**: This report file.

*Note: The dataset is not included in the repository to keep it lightweight. Please see the Dataset section below for instructions on how to download it.*

---

### Dataset 🖼️

The image dataset is not archived directly in this repository. It is available for download from the **"Releases"** section of this project.

#### Instructions to download the data:

1.  **Go to the Releases page:** Click on the link below to access the list of releases:  
    ➡️ **[Go to Project Releases](https://github.com/ricca200xx/Scratch-CNN-Image-Recognition/releases)** 

2.  **Download the archive:** Look for the most recent release (e.g., `v1.0`) and download the `dataset.zip` file from the "Assets" section.

3.  **Unzip the file:** Extract the contents of `dataset.zip` into the main (root) folder of your project.

This will automatically create the `/data` folder with the `train_set` and `test_set`, ready to be used by the notebook.

---

### Model Architecture & Training 🛠️

The model is a **custom-coded Convolutional Neural Network (CNN)** built with PyTorch.

*(**Your action here:** Briefly describe your CNN. For example: "The architecture consists of three convolutional layers with ReLU activation functions, followed by max-pooling layers. After flattening, two fully-connected layers lead to the final output.")*

The model was trained from scratch using the following methods:

-   **Framework:** PyTorch
-   **Optimizer:** *(e.g., Adam, SGD)*
-   **Loss Function:** *(e.g., Cross-Entropy Loss)*
-   **Best Practices:** The model was trained using early stopping and regularization (Dropout) to prevent overfitting.

For a complete implementation, please see the [**Jupyter Notebook**](./notebooks/images_clasification_model.ipynb).

---

### How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ricca200xx/Scratch-CNN-Image-Recognition.git
    cd Scratch-CNN-Image-Recognition
    ```

2.  **Download the dataset:** Follow the instructions in the **Dataset** section above to download and unzip the data.

3.  **Install dependencies:**
    *(It is recommended to create a `requirements.txt` file)*
    ```bash
    pip install torch torchvision pandas matplotlib jupyter
    ```

4.  **Open and run the notebook:**
    Navigate to the `/notebooks` directory and open `images_clasification_model.ipynb` in Jupyter Lab or Notebook to see the complete process of training and evaluation.
