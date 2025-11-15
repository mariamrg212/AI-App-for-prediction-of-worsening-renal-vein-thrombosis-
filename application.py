import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediction of worsening kidney function in renal vein thrombosis ", layout="centered")

# --- Data Loading and Cleanup ---
try:
    # 1. تحميل ملف البيانات الفعلي (يرجى التأكد من أن هذا هو اسم ملفك الصحيح)
    data_df = pd.read_csv('RVT total cases final.csv')
    st.sidebar.success("تم تحميل البيانات الأصلية بنجاح.")

    # 2. تحديد الأعمدة المطلوبة للنموذج (9 أعمدة)
    REQUIRED_FEATURES_CODES = ['NS', 'CRP', 'S.Albumin', 'DM', 'Creat.before', 'Duplex2', 'Treatment', 'D.dimer', 'Sofa.score']
    
    # التأكد من وجود الأعمدة المطلوبة
    data_df = data_df[REQUIRED_FEATURES_CODES]

    # 3. إعادة تسمية الأعمدة
    RENAME_MAP = {
        'NS': 'Nephrotic Syndrome',
        'CRP': 'C-Reactive Protein',
        'S.Albumin': 'Serum Albumin',
        'DM': 'Diabetes Mellitus',
        'Creat.before': 'Serum Creatinine at Admission',
        'Duplex2': 'Thrombus Clearance (Follow-up Duplex)',
        'Treatment': 'Treatment Modality',
        'D.dimer': 'D-dimer',
        'Sofa.score': 'Sofa Score'
    }
    data_df = data_df.rename(columns=RENAME_MAP)
    
    # تحويل الأعمدة الرقمية وملء القيم المفقودة
    for col in data_df.columns:
        if data_df[col].dtype == 'object':
            try:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            except:
                pass
    
    data_df = data_df.fillna(data_df.median(numeric_only=True))

except FileNotFoundError:
    st.error("خطأ: لم يتم العثور على ملف البيانات 'RVT total cases final.csv.csv'.")
    st.stop()
except KeyError as e:
    st.error(f"خطأ: العمود المطلوب {e} غير موجود في ملف البيانات.")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء معالجة ملف البيانات: {e}")
    st.stop()

# --- Model Loading (يجب استبدال هذا النموذج بنموذجك المفضل مثل XGBoost) ---
MODEL_FILE = 'Naive Bayes Model.pkl'

try:
    model = joblib.load(MODEL_FILE)
    st.sidebar.success(f"تم تحميل النموذج '{MODEL_FILE}' بنجاح.")
except FileNotFoundError:
    st.error(f"خطأ: لم يتم العثور على ملف النموذج '{MODEL_FILE}'.")
    st.stop() 
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
    st.stop()
# --- End of Model Loading ---


st.title('Prediction of worsening kidney function in renal vein thrombosis')
st.markdown("---")
st.write("This application predicts the outcome (e.g.,Prediction of worsening kidney function in renal vein thrombosis.")

st.sidebar.header("Patient Input Data")
st.sidebar.markdown("Adjust the parameters below to get a prediction.")

# Dictionary to hold all input data
input_features = {}

# --- Define the order of features for prediction (Must match training order) ---
FINAL_FEATURE_ORDER = [
    'Nephrotic Syndrome', 'C-Reactive Protein', 'Serum Albumin', 
    'Diabetes Mellitus', 'Serum Creatinine at Admission', 
    'Thrombus Clearance (Follow-up Duplex)', 'Treatment Modality', 
    'D-dimer', 'Sofa Score'
]

# Helper function to get median for default value
def get_median(feature_name):
    return data_df[feature_name].median()

# --- Input Widgets for 9 Features ---

# 1. Sofa Score (Continuous/Integer)
sofa_median = int(get_median('Sofa Score'))
input_features['Sofa Score'] = st.sidebar.slider(
    '1. SOFA Score',
    min_value=0, max_value=24, value=sofa_median, step=1
)

# 2. Serum Creatinine at Admission (Continuous)
creat_median = get_median('Serum Creatinine at Admission')
input_features['Serum Creatinine at Admission'] = st.sidebar.number_input(
    '2. Serum Creatinine at Admission (mg/dL)',
    min_value=0.1, max_value=20.0, value=creat_median, step=0.1, format="%.2f"
)

# 3. D-dimer (Continuous)
dimer_median = get_median('D-dimer')
input_features['D-dimer'] = st.sidebar.number_input(
    '3. D-dimer (ng/mL) (if applicable)',
    min_value=0.0, max_value=10000.0, value=dimer_median, step=50.0, format="%.0f"
)

# 4. C-Reactive Protein (Continuous)
crp_median = get_median('C-Reactive Protein')
input_features['C-Reactive Protein'] = st.sidebar.number_input(
    '4. C-Reactive Protein (CRP) (mg/L)',
    min_value=0.0, max_value=500.0, value=crp_median, step=1.0, format="%.1f"
)

# 5. Serum Albumin (Continuous)
albumin_median = get_median('Serum Albumin')
input_features['Serum Albumin'] = st.sidebar.number_input(
    '5. Serum Albumin (g/dL)',
    min_value=1.0, max_value=5.0, value=albumin_median, step=0.1, format="%.2f"
)

# 6. Diabetes Mellitus (Binary)
input_features['Diabetes Mellitus'] = st.sidebar.checkbox("6. Diabetes Mellitus (DM)", False)

# 7. Nephrotic Syndrome (Binary)
input_features['Nephrotic Syndrome'] = st.sidebar.checkbox("7. Nephrotic Syndrome (NS)", False)

# =========================================================================
# 8. Thrombus Clearance (Follow-up Duplex) - NEW IMPLEMENTATION (0 or 1)
# =========================================================================
clearance_options = {'Yes: Partial or complete clearance': 1, 'No: no thrombus clearance': 0}
selected_clearance = st.sidebar.radio(
    "8. Thrombus Clearance (Follow-up Duplex)",
    options=list(clearance_options.keys()),
    index=0 # Default to Yes (1)
)
input_features['Thrombus Clearance (Follow-up Duplex)'] = clearance_options[selected_clearance]


# =========================================================================
# 9. Treatment Modality - NEW IMPLEMENTATION (1 or 2)
# =========================================================================
treatment_options_map = {
    '1: Anticoagulation alone': 1,
    '2: Mechanical thrombectomy plus anticoagulation': 2
}
selected_treatment = st.sidebar.radio(
    "9. Treatment Modality",
    options=list(treatment_options_map.keys()),
    index=0 # Default to 1: Anticoagulation alone
)
input_features['Treatment Modality'] = treatment_options_map[selected_treatment]


st.markdown("---")
# --- Prediction Logic ---

# بناء المصفوفة بنفس الترتيب الذي تم تدريب النموذج عليه
input_list = [
    input_features[feature] for feature in FINAL_FEATURE_ORDER
]

# تحويل القيم المنطقية (True/False) إلى 1/0
final_input_array_list = [
    (1 if item else 0) if isinstance(item, bool) else item 
    for item in input_list
]

input_array = np.array([final_input_array_list])

# Ensure the array is 2D for prediction
input_array = input_array.reshape(1, -1)

if st.sidebar.button('Predict RVT Outcome'):
    try:
        # Perform prediction
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        # تفسير النتائج (نفترض أن 1 هو النتيجة الإيجابية/المرغوبة)
        result_label = "Positive Outcome/Recovery"
        non_result_label = "Negative Outcome/Dependence"
        
        result = result_label if prediction == 1 else non_result_label
        
        # عادةً، العمود [1] يمثل احتمالية الفئة 1
        positive_proba = prediction_proba[1] * 100 
        negative_proba = prediction_proba[0] * 100

        st.markdown("## Prediction Result")
        
        if prediction == 1:
            st.success(f"**Prediction: {result}**")
            st.balloons()
        else:
            st.error(f"**Prediction: {result}**")
            
        st.markdown(f"The model predicts a **{positive_proba:.1f}% chance** of the {result_label} outcome.")

        st.markdown("### Probability Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{result_label}", f"{positive_proba:.1f}%")
        with col2:
            st.metric(f"{non_result_label}", f"{negative_proba:.1f}%")
            
        st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure the input data order and model training features match exactly.")

# Displaying inputs for verification
st.sidebar.markdown("---")
st.sidebar.caption("Input Array Order (Must match model training):")
st.sidebar.code(FINAL_FEATURE_ORDER)