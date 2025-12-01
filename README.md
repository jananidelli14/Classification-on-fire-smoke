# Fire–Smoke–Normal Image Classification using Transfer Learning


###  Project Objective

Build a baseline classification model to identify three classes in images:

* **Fire** — visible flames present
* **Smoke** — smoke visible without clear flames
* **Normal** — no fire or smoke present

This classification task is the first stage before performing object detection using YOLO.

---

##  Dataset Description

* Base dataset downloaded from **Roboflow Fire & Smoke detection dataset**.
* Original dataset contained only **Fire** and **Smoke** images.
* Manually added ~50 **Normal** (background) images: traffic, buildings, roads, etc.

Final folder structure used in Kaggle:

```text
classification_data/
│
├── train/
│   ├── Fire/
│   ├── Smoke/
│   └── Normal/
│
├── valid/
│   ├── Fire/
│   ├── Smoke/
│   └── Normal/
│
└── test/
    ├── Fire/
    ├── Smoke/
    └── Normal/
```

Approximate dataset size:

* **Train:** 9194 images
* **Valid:** 877 images
* **Test:** 440 images

---

##  Why Transfer Learning (ImageNet Weights)?

We use **ImageNet-pretrained weights** because:

* ImageNet has ~1.2M diverse images and 1000 classes.
* Models trained on ImageNet learn strong general visual features: edges, textures, colors, shapes.
* We can fine-tune only the final classification layer for our three classes.

Benefits:

* ✅ Faster training
* ✅ Higher accuracy with limited data
* ✅ Less risk of overfitting compared to training from scratch

---

##  Model Architectures Evaluated

We fine-tuned four convolutional neural network architectures:

1. **ResNet18**

   * Residual connections, relatively small
   * Strong baseline and commonly used as a backbone

2. **VGG16**

   * Deeper network, more parameters
   * Good feature extractor but heavier and slower

3. **MobileNetV2** ⭐

   * Lightweight, designed for mobile/edge devices
   * Depthwise separable convolutions → fewer parameters, faster

4. **EfficientNet-B0**

   * Balanced network with compound scaling
   * Good trade-off between accuracy and efficiency

Each model was initialized with **ImageNet-pretrained weights**, and the final layer was replaced with a new fully connected layer of size 3 (`Fire`, `Smoke`, `Normal`).

---

## 🏋 Training Configuration

* **Input size:** 224×224
* **Batch size:** 16
* **Epochs:** 6
* **Optimizer:** Adam (learning rate = 1e-4)
* **Loss function:** CrossEntropyLoss
* **Scheduler:** StepLR (gamma = 0.5, step_size = 4)
* **Device:** GPU (CUDA on Kaggle)

### Data Augmentation

For the training set, we applied:

* RandomResizedCrop
* RandomHorizontalFlip
* RandomRotation
* ColorJitter
* Normalization (ImageNet mean & std)

Validation and test sets only used resize, center crop, and normalization.

---

## 📊 Results

Validation and test accuracies for each model:

* **ResNet18** → Val: 93.27% | Test: 94.32%
* **VGG16** → Val: 92.36% | Test: 92.73%
* **EfficientNet-B0** → Val: 92.93% | Test: 93.18%
* **MobileNetV2** → **Val: 93.39% | Test: 95.00%** ✅

### ✅ Best Model: MobileNetV2

MobileNetV2 achieved the **highest validation and test accuracy** while remaining lightweight and fast. Therefore, we selected **MobileNetV2** as the final classification model for this phase.

---

##  Qualitative Testing

We also tested the final MobileNetV2 model on sample images from the test set and custom images (e.g., road and forest scenes, buildings, etc.).

* Correct predictions: images with flames → `Fire`, smoke-only images → `Smoke`, clean backgrounds → `Normal`.
* Observed limitation: some highly red/orange backgrounds (e.g., red trees) were misclassified as `Fire`. This indicates a **color bias** in the dataset, since most fire images are very red/orange.

Mitigation idea:

* Add more `Normal` images that contain strong red/orange regions.
* Fine-tune the model again to reduce false positives.

---

##  Role of Classification in the Overall Project

This classification model is a **baseline stage** before object detection:

* It verifies that a deep learning model can **distinguish Fire vs Smoke vs Normal** using our dataset.
* Helps check dataset quality and class separability.
* Faster and simpler than directly starting with YOLO detection.

Once classification performance is satisfactory (as above), the next step is to:

* Train **YOLO11** for fire and smoke **object detection** using the same dataset.
* Use the detection model for localisation (where in the image fire/smoke appears) and real-time monitoring.

---

## Repository Contents 

* `classification_training.ipynb` — Kaggle notebook with full pipeline.
* `normal_images_backup.zip` — manually collected Normal-class screenshots (traffic, buildings, etc.).
* `README.md` — project documentation (this file).

---

##  Summary

* Built a 3-class image classifier for **Fire / Smoke / Normal**.
* Used **ImageNet-pretrained CNNs** with transfer learning.
* Evaluated four architectures; **MobileNetV2** performed best with ~93% validation and 95% test accuracy.
* This provides a strong baseline and confirms that the dataset is suitable for fire/smoke recognition.
