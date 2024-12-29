import os
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyD5wDuk2saq90X2Tdvmby9bQ-XKtZwYm-U"
import  google.generativeai as geni
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import classifier functions
from models.AbdominalTraumaDetection import AbdominalTraumaDetection
from models.KidneyDiseasesClassification import KidneyDiseasesClassification
from models.ChestXRay import ChestXRay
from models.BoneFractures import BoneFractures
from models.Knee_Osteoporosis import Knee_Osteoporosis

def infer_abdominal_trauma(image_path):
    model_path = r"D:\torch\cnn_proj\models\model\efficientnet_b0_Abdominal_Trauma_Detection.pth"
    trauma_detector = AbdominalTraumaDetection(model_path=model_path, threshold=0.7)
    result = trauma_detector.infer(image_path)
    description = (
        "Detects and classifies abdominal trauma using an EfficientNet-based model. "
        "The dataset consists of CT scan images providing ground truth labels for injuries. It identifies potential injuries such as Bowel Injury, Extravasation Injury, Kidney Injury (Healthy, Low, High), Liver Injury (Healthy, Low, High), and Spleen Injury (Healthy, Low, High). "
        "Output includes a prediction dictionary indicating presence (1) or absence (0) of injuries and confidence scores for each prediction."
    )
    return result, description

def infer_kidney_diseases(image_path):
    model_path = r"D:\torch\cnn_proj\models\model\Kidney_Diseases_Classfication_Model.pth"
    classifier = KidneyDiseasesClassification(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Classifies various kidney diseases based on the provided medical image. "
        "The dataset was collected from hospital-based PACS focusing on kidney-related diagnoses such as tumor, cyst, normal, or stone. "
        "It consists of four classes and requires high precision in classification to assist accurate medical diagnoses."
    )
    return result, description

def infer_chest_xray(image_path):
    model_path = r"D:\torch\cnn_proj\models\model\chest_xray.pth"
    classifier = ChestXRay(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Analyzes chest X-rays to detect abnormalities such as infections or diseases. "
        "The dataset focuses on pneumonia, an infection inflaming the air sacs in the lungs and a leading cause of death among children under 5. "
        "Images are labeled as (disease: NORMAL/BACTERIA/VIRUS)-(randomized patient ID)-(image number of a patient). "
        "The model aims for precise classification to assist in accurate diagnoses."
    )
    return result, description

def infer_bone_fractures(image_path):
    model_path = r"D:\torch\cnn_proj\models\model\Bone_Fracture_Binary_Classification.pth"
    classifier = BoneFractures(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Identifies bone fractures using binary classification. "
        "The dataset includes fractured and non-fractured X-ray images from all anatomical body regions, including lower extremity, upper extremity, lumbar, hips, and knees. "
        "Timely and accurate diagnosis is critical to mitigate risks associated with delayed treatments."
    )
    return result, description


def infer_knee_osteoporosis(image_path):
    model_weights_path = r"D:\torch\cnn_proj\models\model\Knee_model_weights.pth"
    classifier = Knee_Osteoporosis()
    result = classifier.inf(model_weights_path, image_path)
    description = (
        "Classifies knee images to identify osteoporosis. "
        "The dataset is categorized into Normal (no signs of osteoporosis), Osteopenia (early stages of bone density loss), and Osteoporosis (advanced bone density degradation). "
        "The output includes the predicted class and confidence scores, supporting timely detection and treatment."
    )
    return result, description


def generate_radiology_report(classifier_description, classifier_outputs, patient_name, patient_age, patient_gender, date):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

    template = """
    You are an expert radiology report generator specializing in X-ray interpretations. 
    Using the input provided below, create a detailed and professional X-ray report that includes:
    1. A description of the classifier results.
    2. Relevant medical recommendations based on the classifier outputs.
    3. If the condition is negative (indicating a disease), include a note about potential complications or outcomes if the disease is not treated.

    Input details:
    - Classifier Function: {classifier_description}
    - Classifier Results: {classifier_outputs}
    - Patient Name: {patient_name}
    - Patient Age: {patient_age}
    - Patient Gender: {patient_gender}
    - Date: {date}

    Generate the report with a clear structure and provide the necessary medical insights.

    Expected Report Structure:
    - **Patient Information**: Include name, age, gender, and date.
    - **Findings**: Summarize the classifier results.
    - **Impression**: Provide a concise interpretation of the findings.
    - **Recommendations**: Suggest next steps or treatments.
    - **Warnings**: If the condition is negative, highlight what could happen if untreated.
    """

    prompt = PromptTemplate(
        input_variables=[
            "classifier_description",
            "classifier_outputs",
            "patient_name",
            "patient_age",
            "patient_gender",
            "date"
        ],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    report = chain.run(
        classifier_description=classifier_description,
        classifier_outputs=classifier_outputs,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        date=date
    )

    return report

# Streamlit app
st.title("X-ray Report Generator")

# Upload image
uploaded_image = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

# Select classifier
classifier_options = {
    "Abdominal Trauma": infer_abdominal_trauma,
    "Kidney Diseases": infer_kidney_diseases,
    "Chest X-ray": infer_chest_xray,
    "Bone Fractures": infer_bone_fractures,
    "Knee Osteoporosis": infer_knee_osteoporosis
}

selected_classifier = st.selectbox("Select Classifier", list(classifier_options.keys()))

# Patient information
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)
patient_gender = st.selectbox("Patient Gender", ["Male", "Female"])
date = st.date_input("Examination Date")

if st.button("Generate Report"):
    if uploaded_image and patient_name and patient_age and patient_gender:
        # Save uploaded image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.read())

        # Run the selected classifier
        infer_function = classifier_options[selected_classifier]
        result, description = infer_function("temp_image.jpg")

        # Generate the report
        report = generate_radiology_report(
            classifier_description=description,
            classifier_outputs=result,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            date=date
        )

        # Display the report
        st.subheader("Generated X-ray Report")
        st.markdown(report)
    else:
        st.error("Please complete all fields and upload an image.")
