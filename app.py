import gradio as gr
import pandas as pd
import pickle
import numpy as np

# model load
try:
    with open("water_potability_predict.pkl", "rb") as f:
        load_model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'water_potability_predict.pkl'")

# WHO standard 
WHO_STANDARDS = {
    'ph': {'min': 6.5, 'max': 8.5, 'unit': '', 'label': 'pH'},
    'Hardness': {'min': 0, 'max': 300, 'unit': 'mg/L', 'label': 'Hardness'},
    'Solids': {'min': 0, 'max': 1000, 'unit': 'mg/L', 'label': 'Solids/TDS'},
    'Chloramines': {'min': 0, 'max': 4, 'unit': 'mg/L', 'label': 'Chloramines'},
    'Sulfate': {'min': 0, 'max': 250, 'unit': 'mg/L', 'label': 'Sulfate'},
    'Conductivity': {'min': 0, 'max': 1400, 'unit': 'μS/cm', 'label': 'Conductivity'},
    'Organic_carbon': {'min': 0, 'max': 10, 'unit': 'mg/L', 'label': 'Organic Carbon'},
    'Trihalomethanes': {'min': 0, 'max': 100, 'unit': 'μg/L', 'label': 'Trihalomethanes'},
    'Turbidity': {'min': 0, 'max': 5, 'unit': 'NTU', 'label': 'Turbidity'}
}

feature_names = list(WHO_STANDARDS.keys())

# prediction
def predict_potability(ph, Hardness, Solids, Chloramines, Sulfate, 
                       Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    
    input_df = pd.DataFrame([[ph, Hardness, Solids, Chloramines, Sulfate, 
                              Conductivity, Organic_carbon, Trihalomethanes, Turbidity]],
                            columns=feature_names)
    
    probs = load_model.predict_proba(input_df)[0]
    safe_prob = probs[1]
    unsafe_prob = probs[0]
    
    
    best_threshold = 0.52
    prediction = 1 if safe_prob >= best_threshold else 0

    compliance_text = ""
    compliant_count = 0
    
    for feat in feature_names:
        val = locals()[feat]
        std = WHO_STANDARDS[feat]
        is_ok = std['min'] <= val <= std['max']
        if is_ok: compliant_count += 1
        
        icon = "✅" if is_ok else "❌"
        compliance_text += f"{icon} {std['label']}: {val:.2f} {std['unit']} (Limit: {std['min']}-{std['max']})\n"

    res_icon = "🟢 SAFE" if prediction == 1 else "🔴 UNSAFE"
    
    final_output = f"【 1. MACHINE LEARNING PREDICTION 】\n"
    final_output += f"FINAL RESULT: {res_icon}\n"
    final_output += f"Confidence: {max(safe_prob, unsafe_prob)*100:.1f}%\n\n"

    final_output += f"【 2. PROBABILITY ANALYSIS 】\n"
    final_output += f"Safe Probability   : {safe_prob*100:.1f}%\n"
    final_output += f"Unsafe Probability : {unsafe_prob*100:.1f}%\n"
    final_output += f"Threshold Used     : {best_threshold}\n\n"

    final_output += f"【 3. WHO STANDARD COMPARISON 】\n"
    final_output += f"Compliance: {compliant_count}/9 Parameters match WHO standards.\n"
    final_output += "-" * 45 + "\n"
    final_output += compliance_text
    
    return final_output


inputs = [
    gr.Slider(0, 14, label="🧪 pH", value=7.0),
    gr.Slider(0, 500, label="💎 Hardness", value=150),
    gr.Slider(0, 30000, label="🧂 Solids", value=15000),
    gr.Slider(0, 15, label="🧬 Chloramines", value=4.0),
    gr.Slider(0, 500, label="⚗️ Sulfate", value=250),
    gr.Slider(0, 2000, label="⚡ Conductivity", value=500),
    gr.Slider(0, 30, label="🌿 Organic Carbon", value=10),
    gr.Slider(0, 150, label="🧪 Trihalomethanes", value=60),
    gr.Slider(0, 10, label="🌊 Turbidity", value=3.0)
]


demo = gr.Interface(
    fn=predict_potability,
    inputs=inputs,
    outputs=gr.Textbox(label="Analysis Results", lines=18),
    title="💧 Water Potability AI Analyzer",
    description="XGBoost model and WHO Standared prediction",
    flagging_mode="never" # allow_flagging এর বদলে flagging_mode="never" ব্যবহার করুন
)

if __name__ == "__main__":
    
    demo.launch()






























