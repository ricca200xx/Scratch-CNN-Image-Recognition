# Custom CNN for Image Classification from Scratch

A hands-on project building a Convolutional Neural Network from the ground up to classify images into 20 distinct categories using PyTorch.

### Project Overview 🧠

This project covers the end-to-end development of a deep learning model for multi-class image classification. The primary objective is to design, build, and train a **Convolutional Neural Network (CNN)** from scratch, emphasizing custom architecture design and robust training practices without relying on pre-trained models.

---

### Repository Structure 📂

The project is organized with the following files and directories:

* `├──` **`/data`**: This directory is intended to hold the image **dataset** after it has been downloaded from the Releases page. It is intentionally empty in the repository and is listed in `.gitignore`.

* `├──` **`images_clasification_model.ipynb`**: The core of the project. This **Jupyter Notebook** contains the entire workflow, from data loading and preprocessing to model architecture definition, training, and evaluation.

* `├──` **`submission.csv`**: An example **submission file** in the required format, showing the model's predictions on the test set.

* `├──` **`README.md`**: This documentation file.

* `├──` **`requirements.txt`**: A list of the **Python dependencies** required to run the notebook. Install them using `pip install -r requirements.txt`.

* `└──` **`.gitignore`**: A configuration file that tells Git which files and folders to **ignore**, such as the `/data` directory and Python cache files.

---

### Dataset 🖼️

The dataset is not stored directly in this repository to keep it lightweight. It can be downloaded from the project's **Releases** page.

#### How to Download the Data:

1.  **Go to the Releases Page:** Click the link below to access the project's releases:  
    ➡️ **[Go to Project Releases](https://github.com/ricca200xx/Scratch-CNN-Image-Recognition/releases)**

2.  **Download the Archive:** Find the latest release (e.g., `v1.0`) and download the `dataset.zip` file from the "Assets" section.

3.  **Unzip the File:** Extract the contents of `dataset.zip` into the project's root directory. This will create the `/data` folder containing the `train_set` and `test_set`, ready to be used by the notebook.

---

### Model Architecture & Training 🛠️

The model is a **custom Convolutional Neural Network (CNN)** designed and implemented in PyTorch. The architecture is logically separated into two main components: a **feature extractor** composed of convolutional layers and a **classifier** made of fully-connected layers.

#### Architectural Breakdown 🧠

The network processes `40x40x3` input images through the following layers:

1.  **Feature Extractor (Convolutional Base):** This part of the network is responsible for identifying low-level and high-level features (edges, textures, shapes) in the images. It consists of three sequential blocks, each containing:
    * **Convolutional Layer:** Applies a set of filters to detect features.
    * **ReLU Activation:** Introduces non-linearity, allowing the model to learn more complex patterns.
    * **Max-Pooling Layer:** Down-samples the feature map, reducing dimensionality and making the model more robust to variations in feature positions.

2.  **Classifier (Fully-Connected Head):** After feature extraction, the 2D feature maps are flattened into a 1D vector and fed into the classifier:
    * **Flatten Layer:** Transitions from the convolutional base to the dense layers.
    * **Fully-Connected Layer 1 (Dense):** A standard neural network layer that learns combinations of the features extracted by the convolutional base.
    * **Dropout:** A regularization technique that randomly sets a fraction of input units to 0 during training to prevent overfitting.
    * **Fully-Connected Layer 2 (Output):** The final layer that produces the raw scores (logits) for each of the 20 classes.

#### Training Protocol ⚙️

The model was trained from scratch with the following configuration:

* **Framework:** **PyTorch**
* **Optimizer:** **Adam** (Adaptive Moment Estimation), an efficient optimization algorithm that adapts the learning rate for each parameter, making it well-suited for computer vision tasks.
* **Loss Function:** **Cross-Entropy Loss**, the industry-standard loss function for multi-class classification problems.
* **Key Training Strategies:**
    * **Early Stopping:** The training process was monitored using a validation set. Training was halted when the validation loss stopped improving to save the model at its point of peak performance.
    * **Regularization:** Dropout was used to prevent neuron co-adaptation and improve the model's ability to generalize to unseen data.

For a line-by-line implementation and to see the exact hyperparameters used, please review the [**Jupyter Notebook**](./images_clasification_model.ipynb).

---

### How to Run the Code 🚀

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ricca200xx/Scratch-CNN-Image-Recognition.git
    cd Scratch-CNN-Image-Recognition
    ```

2.  **Download the dataset:** Follow the instructions in the **Dataset** section above.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the notebook:**
    Open the `images_clasification_model.ipynb` file in Jupyter Lab or a similar environment to execute the code and see the complete workflow.
