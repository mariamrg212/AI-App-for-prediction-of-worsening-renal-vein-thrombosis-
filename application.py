import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediction of worsening kidney function in renal vein thrombosis", layout="centered")

# --- Data Loading and Cleanup ---
try:
    # 1. تحميل ملف البيانات الفعلي
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
    
    # تحويل جميع الأعمدة إلى أرقام قسراً لتحاشي أخطاء الـ string
    for col in data_df.columns:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    # ملء القيم المفقودة بالوسيط لكل عمود
    data_df = data_df.fillna(data_df.median(numeric_only=True))

except FileNotFoundError:
    st.error("خطأ: لم يتم العثور على ملف البيانات 'RVT total cases final.csv'.")
    st.stop()
except KeyError as e:
    st.error(f"خطأ: العمود المطلوب {e} غير موجود في ملف البيانات.")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء معالجة ملف البيانات: {e}")
    st.stop()

MODEL_FILE = 'Naive Bayes Model.pkl'

try:
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    st.error(f"خطأ: لم يتم العثور على ملف النموذج '{MODEL_FILE}'.")
    st.stop() 
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل النموذج: {e}")
    st.stop()
# --- End of Model Loading ---

st.title('Prediction of worsening kidney function in renal vein thrombosis')
st.markdown("---")
st.write("This application predicts the outcome (e.g., Prediction of worsening kidney function in renal vein thrombosis).")

st.sidebar.header("Patient Input Data")
st.sidebar.markdown("Adjust the parameters below to get a prediction.")

input_features = {}

FINAL_FEATURE_ORDER = [
    'Nephrotic Syndrome', 'C-Reactive Protein', 'Serum Albumin', 
    'Diabetes Mellitus', 'Serum Creatinine at Admission', 
    'Thrombus Clearance (Follow-up Duplex)', 'Treatment Modality', 
    'D-dimer', 'Sofa Score'
]

# Helper function to get median safely
def get_median(feature_name):
    val = data_df[feature_name].median()
    return 0.0 if pd.isna(val) else float(val)

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

# 8. Thrombus Clearance (Follow-up Duplex)
clearance_options = {'Yes: Partial or complete clearance': 1, 'No: no thrombus clearance': 0}
selected_clearance = st.sidebar.radio(
    "8. Thrombus Clearance (Follow-up Duplex)",
    options=list(clearance_options.keys()),
    index=0
)
input_features['Thrombus Clearance (Follow-up Duplex)'] = clearance_options[selected_clearance]

# 9. Treatment Modality
treatment_options_map = {
    '1: Anticoagulation alone': 1,
    '2: Mechanical thrombectomy plus anticoagulation': 2
}
selected_treatment = st.sidebar.radio(
    "9. Treatment Modality",
    options=list(treatment_options_map.keys()),
    index=0
)
input_features['Treatment Modality'] = treatment_options_map[selected_treatment]

st.markdown("---")

# --- Prediction Logic ---
input_list = [input_features[feature] for feature in FINAL_FEATURE_ORDER]
final_input_array_list = [
    (1 if item else 0) if isinstance(item, bool) else item 
    for item in input_list
]

input_array = np.array([final_input_array_list]).reshape(1, -1)

if st.sidebar.button('Predict RVT Outcome'):
    try:
        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0]
        
        result_label = "Probability of worsening kidney function"
        non_result_label = "Probability of no worsening kidney function"
        result = result_label if prediction == 1 else non_result_label
        
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

# Upload Section for Batch Prediction
st.sidebar.markdown("---")
st.sidebar.caption("Input Array Order (Must match model training):")
st.sidebar.code(FINAL_FEATURE_ORDER)

uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type=["csv"])

if uploaded_file is not None:
    try:
        data_file = pd.read_csv(uploaded_file)
        data_file = data_file[REQUIRED_FEATURES_CODES]
        data_file = data_file.rename(columns=RENAME_MAP)
        
        for col in data_file.columns:
            data_file[col] = pd.to_numeric(data_file[col], errors='coerce')
        
        data_file = data_file.fillna(data_file.median(numeric_only=True))

        st.write("Uploaded File Preview:")
        st.dataframe(data_file.head())
        
        if st.button("Predict Batch"):
            # إعادة ترتيب الأعمدة لتطابق الترتيب النهائي للنموذج
            batch_input = data_file[FINAL_FEATURE_ORDER]
            file_prediction = model.predict(batch_input)
            data_file['prediction'] = file_prediction
            
            st.success("Batch Prediction Completed Successfully!")
            st.dataframe(data_file.head())
            
            csv_result = data_file.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV Predictions",
                data=csv_result,
                file_name="predictions_result.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
