# Smart X-Ray System for Automated Medical Diagnosis and Reporting

## OVERVIEW
The Smart X-Ray System represents a breakthrough in medical diagnostics by harnessing the power of Artificial Intelligence (AI) to automatically analyze X-ray images and generate comprehensive diagnostic reports. This innovative solution addresses key challenges in radiology, such as managing workloads, ensuring diagnostic accuracy, and improving accessibility in underserved regions. By combining machine learning for precise image analysis with a large language model (LLM) for generating professional reports, the system offers a complete, integrated tool that enhances healthcare outcomes and streamlines medical workflows.

---

## ARCHITECTURE

### 2.1 INPUT: DATASETS
The system processes the following X-ray datasets:
- **Abdominal Trauma Dataset** (RSNA 2023 Abdominal Trauma Detection Dataset)
- **Chest X-ray Dataset** (Labeled Chest X-ray Images Dataset)
- **Kidney Diseases Dataset** (CT Kidney Dataset)
- **Bone Fractures Dataset** (Fracture Multi-region X-ray Dataset)
- **Knee Osteoporosis Dataset** (Multi-class Knee Osteoporosis Dataset)

Each dataset is used to train a dedicated model for its specific diagnostic task.

---

### 2.2 PREPROCESSING AND ANALYSIS: NOTEBOOKS
The following notebooks handle preprocessing and analysis:
- `AbdominalTraumaDetection.py`: Processes the Abdominal Trauma dataset using EfficientNet.
- `BoneFractures.py`: Detects fractures across multiple regions.
- `ChestXRay.py`: Classifies conditions (Normal, Bacterial Pneumonia, Viral Pneumonia).
- `KidneyDiseasesClassification.py`: Identifies conditions like stones, cysts, and tumors.
- `Knee_Osteoporosis.py`: Categorizes osteoporosis levels (Normal, Early, Advanced).

---

### 2.3 MODELS
Each notebook produces a trained model:
- **Abdominal Trauma Detection**: `efficientnet_b0_Abdominal_Trauma_Detection.pth`
- **Bone Fracture Classification**: `Bone_Fracture_Binary_Classification.pth`
- **Chest X-ray Classification**: `chest_xray.pth`
- **Kidney Disease Classification**: `Kidney_Diseases_Classification_Model.pth`
- **Knee Osteoporosis Classification**: `Knee_model_weights.pth`

---

### 2.4 REPORT GENERATION
Predictions from models are processed by an LLM to:
- Translate outputs into diagnostic terms.
- Generate professional reports.
- Include confidence scores and recommendations.

---

### 2.5 USER INTERFACE: STREAMLIT APPLICATION
The Streamlit app features:
- **Upload X-rays**: Users can upload X-ray images.
- **Model Results**: Displays predictions and confidences.
- **Generated Reports**: Provides detailed diagnostic reports.

### System Workflow
![System Workflow](app.gif)

---

### 2.6 WORKFLOW AND PROMPT INTEGRATION
Prompts include parameters such as:
- **Model_desc**: Description of the model.
- **Model_out**: Model predictions and confidences.
- **Patient Details**: Name, Age, Gender, and Date.

---

### 2.7 Architecture Flow
1. **Input**: Abdominal Trauma, Chest X-ray, Kidney Diseases, Bone Fractures, Knee Osteoporosis datasets.
2. **Preprocessing**: Using dedicated Python notebooks.
3. **Model Training/Inference**: Efficient models for each dataset.
4. **Report Generation**: LLMs for patient-specific diagnostic reports.
5. **User Interface**: Streamlit app for user interaction.

---

## TEAM
1. [Ahmed Selim Mahmoud](https://www.linkedin.com/in/ahmed-selim-mahmoud/)
2. [Sohila Ayman Lashien](https://www.linkedin.com/in/sohila-lashien-0b31462a2/)
3. [Abdelrahman Elsayed Mohammed](https://www.linkedin.com/in/abdoelsayed/)
4. [Hager Mohammed Hegazy](https://www.linkedin.com/in/hager-hagezy-4253a4250?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
5. [Nour Ali Abdelmordi Saif](https://www.linkedin.com/in/ACoAAEKlKlMBMYRSMzKFxNyVDliYxfSmYzpgfoQ?lipi=urn%3Ali%3Apage%3Ad_flagship3_detail_base%3BW45yY8MQRi%2BvH2ZM9tP6tw%3D%3D)
6. [Mennat-Allah Medhat Mostafa](https://www.linkedin.com/in/menna-medhat-64058a29a/)
7. [Mennat-Allah Yousri Rabeh](https://www.linkedin.com/in/menna-allah-yousri-6a7900288?miniProfileUrn=urn%3Ali%3Afs_miniProfile%3AACoAAEXmr98BJfJRLtIb7uF86PugvOEqaCMwLL4&lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_all%3BqB7lghllRaa4qpH7i8LzWQ%3D%3D)

---

## NOTEBOOKS

### Abdominal-Trauma-Detection
#### Overview
Focuses on binary multi-label classification for identifying various injuries (e.g., bowel, kidney, liver).  

#### Dataset
15,632 CT scans of 3,147 patients.  

#### Results
- **Training Accuracy**: 89.21%
- **Validation Accuracy**: 89.05%
- **Test Accuracy**: 90.21%

---

### Chest X-ray Classification
#### Overview
Classifies chest X-rays into Normal, Bacterial Pneumonia, and Viral Pneumonia.  

#### Dataset
Publicly available dataset of labeled chest X-rays.  

#### Results
Detailed results and challenges are addressed in the corresponding notebook.

---

### Knee Osteoporosis Detection
#### Overview
Classifies stages of knee osteoporosis into Normal, Osteopenia, and Osteoporosis.  

#### Results
- **Accuracy**: 83.28%
- **F1-Score**: 83.48%

---

### Kidney Disease Classification
#### Overview
Detects cysts, stones, and tumors in kidney CT scans.  

#### Results
- **Accuracy**: 99%
- **F1-Score**: 99.1%

---

### Bone Fractures Classification
#### Overview
Detects bone fractures across multiple regions.  

#### Results
- Detailed classification accuracy for each anatomical region.

---

## Contributions
- **AI Models**: Fine-tuned CNN-based architectures.
- **Streamlit Application**: User-friendly interface for diagnosis and reporting.
- **LLM Integration**: Automated report generation for clinicians.

