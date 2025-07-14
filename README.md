🛰️ Classification and Mapping of Agricultural Crops Using Satellite Images and Machine Learning
This platform is a complete system for analyzing and classifying agricultural land using Sentinel-2 satellite imagery, spectral indices, and machine learning algorithms like XGBoost and CNN. It is designed to distinguish between cultivated and non-cultivated land, and to assess crop suitability (e.g., wheat, barley) based on land characteristics.

🎯 Project Objectives
- Classify land as arable (cultivated) or non-arable.

- Use spectral indices (NDVI, EVI, SAVI) to assess vegetation health.

- Predict crop suitability using a trained XGBoost model.

- Segment images into homogeneous regions to improve analysis.

- Provide a user-friendly web platform for visualization and prediction.
  
📁 Project Structure

<img width="887" height="351" alt="image" src="https://github.com/user-attachments/assets/bad89d28-db95-4dc7-b371-85f0450d9801" />

🧠 Technologies Used
Satellite Data: Sentinel-2 (13 bands, multispectral)

Spectral Indices: NDVI, EVI, SAVI, NDWI, etc.

Machine Learning:

XGBoost (for structured/tabular spectral data)

CNN (for RGB classification from bands B2, B3, B4)

Image Processing: Rasterio, NumPy, OpenCV

Frameworks: TensorFlow / Keras, Scikit-learn

Web Interface: HTML/CSS + JavaScript or Flask frontend
📊 Model Performance

| Model   | Accuracy | Validation Loss |
| ------- | -------- | --------------- |
| CNN     | 95.12%   | 0.1671          |
| XGBoost | High     | Interpretable   |


$ => CNN trained on RGB images (B2, B3, B4 bands)

$ => XGBoost trained on vegetation indices + statistical features

🌍 Platform Features

| Feature                | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| 🏠 Home Page           | Landing page of the platform                         |
| 🔍 Predict Class       | Upload image → Predict arable/non-arable land        |
| 🧩 Segment Image       | Apply segmentation to split into homogeneous regions |
| 🌾 Crop Classification | Predict crop suitability (wheat, barley, unknown)    |
| 📈 Vegetation Analysis | Show NDVI/EVI/SAVI overlays                          |
| 📁 Results Export      | Download predictions and analysis reports            |


XGBoost trained on vegetation indices + statistical features

🚀 How to Run
1 -Clone the repo:
  git clone https://github.com/yourusername/agri-crop-classification.git
  cd agri-crop-classification
  
2 -Set up the Python environment:
  pip install -r requirements.txt
  
3 -Run preprocessing & training:
  python src/preprocessing/prepare_data.py
  python src/xgboost_model/train_model.py

4 -Launch the web interface:
  cd web-platform/
  python app.py

🧪 Dataset
Source: EuroSAT (Sentinel-2)

27,000+ image patches (64×64 px) in 10 land cover classes

Used both RGB and Multispectral (.tif) formats

🔍 Screenshots

| Platform Pages                    | Description                  |
| --------------------------------- | ---------------------------- |
| ![Home](assets/homepage.png)      | Main dashboard               |
| ![Segmentation](assets/seg.png)   | Image segmentation view      |
| ![Prediction](assets/predict.png) | Crop suitability predictions |

📜 License

This project is open source and available under the MIT License.

🙌 Acknowledgments

Supervised by: Dr. Said Labed

Supported by: Algerian Space Agency (ASAL) and University Constantine 2

Developed by: Nour Elyakine Ben Aicha & Hamla Ferial

[Classification_and_mappig_of_agricultural_crops_Using_satellite_images_and_machine_learning_algorithms.pdf](https://github.com/user-attachments/files/21212080/Classification_and_mappig_of_agricultural_crops_Using_satellite_images_and_machine_learning_algorithms.pdf)


Home Page

![Home Page](https://github.com/user-attachments/assets/6ba19f28-c32a-441c-abfd-102f13dec053)

Main Page

![main Page](https://github.com/user-attachments/assets/addee02c-396e-43cc-a6d8-09656ba24acb)

Predict Class

![Predict Class](https://github.com/user-attachments/assets/ccfaece2-8d42-4b87-9da6-692e45b01b16)

Crop Classification

![Crop Classification](https://github.com/user-attachments/assets/cd9cb4db-6c19-46fb-8253-db107b8d368c)

Vegetation Analysis

![Vegetation Analysis](https://github.com/user-attachments/assets/b050d478-7d45-4144-a44b-6a31f6bdf5c2)

Segment Image

![Segment Image](https://github.com/user-attachments/assets/47d696eb-6ad9-42cd-9b4d-ec8987ffd1aa)

"You can send me a message if you need anything. Thank you!"
📨 : nourbenaicha90@gmail.com




