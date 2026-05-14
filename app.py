import streamlit as st
import pandas as pd
import joblib
import os
import xgboost as xgb
import re
import plotly.express as px
import base64
import datetime
from fpdf import FPDF
import shap
import matplotlib.pyplot as plt

year = datetime.datetime.now().year

# 1. INITIAL CONFIG
st.set_page_config(layout="wide", page_title="Promotion AI Prediction", page_icon="🚀")

MAIN_BLUE = "#00327D"
LIGHT_BLUE = "#7399C6"
GOLD = "#72383D"
OLIVE = "#3D3E4A"
GREY = "#30313E"

# 2. THEME STYLING
st.markdown(f"""
    <style>
    
    /* Target Sidebar Navigation Links */
    [data-testid="stSidebarNav"] span {{
        font-size: 18px !important;
        font-weight: 350 !important;
    }}

    /* Sidebar Background and Text */
    [data-testid="stSidebar"] {{ background-color: {MAIN_BLUE}; }}
    [data-testid="stSidebar"] * {{ color: white !important; }}

    /* Metric Cards */
    .metric-card {{
        background-color: white; padding: 30px 20px; border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center; margin-bottom: 10px;
    }}

    /* Career Path Training Cards */
    .training-card {{
        background-color: #E8EEF7 !important; 
        padding: 20px; 
        border-radius: 10px;
        min-height: 450px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .outcome-box {{
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 8px;
        border: 1px dashed {OLIVE}; 
        margin-top: auto; 
        font-size: 14px;
        color: {GREY} !important;
    }}

    /* Certification Section Cards */
    .cert-card {{
        background-color: #E4F2FF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #C5DFF8;
        display: flex;
        flex-direction: column;
        min-height: 300px;
        height: 100%;
    }}
    
    /* Path Selection Cards */
    .path-card {{
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        height: 100%;
    }}

    /* NEW: Warning Card Style */
    .cert-warning {{
        background-color: #F9C5C5 !important;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #EBB0B0 !important;
        min-height: 300px;
        height: 100%;
    }}

    /* Milestone Cards */
    .milestone-card {{
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        border: 1px solid #f0f2f6; height: 100%;
    }}

    .outcome-box {{
        background-color: #ffffff; padding: 15px; border-radius: 8px;
        border: 1px dashed {OLIVE}; margin-top: 15px; font-size: 14px;
        color: {MAIN_BLUE};
    }}

    .metric-title {{ color: {OLIVE}; font-size: 14px; font-weight: bold; text-transform: uppercase; }}
    .metric-value {{ color: {GOLD}; font-size: 36px; font-weight: bold; }}

    h1, h2, h3 {{ color: {MAIN_BLUE}; }}
    hr {{ border-top: 2px solid {GOLD}; }}
    .sub-title {{ color: {OLIVE}; font-size: 18px; margin-top: -15px; margin-bottom: 20px; }}

    /* Unified Styling for Selectbox & Text Input */
    div[data-baseweb="select"] > div:focus-within, 
    div[data-baseweb="input"] > div:focus-within,
    .stTextInput div[data-baseweb="input"]:focus-within {{
        border-color: {LIGHT_BLUE} !important;
        box-shadow: 0 0 0 1px {LIGHT_BLUE} !important;
    }}

    /* Custom Refresh Button Styling */
    div[data-testid="stHorizontalBlock"] > div:nth-child(4) button {{
        width: 39px !important;
        height: 39px !important;
        padding: 0px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 8px; 
        margin-top: -16px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# 3. ASSETS & ENGINE
MODEL_PATH = os.path.join('Models', 'promotion_model-2.pkl')
ENCODER_PATH = os.path.join('Models', 'categorical_encoders.pkl')
DATA_PATH = os.path.join('Data', 'combined_promotion_data.csv')
SHAP_PATH = os.path.join('Models', 'shap_values_full_dataset.pkl')

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    shap_vals = joblib.load(SHAP_PATH)
    return model, encoders, shap_vals

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()

    def convert_tenure_to_float(val):
        if pd.isna(val) or not isinstance(val, str): return 0.0
        try:
            y = int(re.search(r'(\d+)y', val).group(1)) if 'y' in val else 0
            m = int(re.search(r'(\d+)m', val).group(1)) if 'm' in val else 0
            return y + (m / 12.0)
        except:
            return 0.0

    if 'No of year in salary grade' in data.columns:
        data['No of year in salary grade'] = data['No of year in salary grade'].apply(convert_tenure_to_float)

    if 'Years_Since_HSE' not in data.columns:
        data['Years_Since_HSE'] = 0.0
    return data

MODEL_FEATURES = [
    'Age', 'Gender', 'Religion', 'Race', 'No of year in salary grade',
    'Position SKG', '2021', '2022', '2023', '2024', '2025', 'Result HSE',
    'PECAS EA / CAP MTG', 'COC Requirement PESITAS',
    'COC Requirement CHARGEMAN', 'CCM Score (TSKG/BSKG)', 'Edge', 'TECH',
    'PPA_Average', 'Years_Until_Retirement', 'IR Case_Clear', 'Edge_Gap',
    'Tech_Gap', 'CCM_Gap', 'Years_Since_HSE',
]

@st.cache_data
def get_predictions(_df, _model, _encoders):
    df_proc = _df.copy()
    for col, le in _encoders.items():
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].astype(str).str.strip().apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    for col in MODEL_FEATURES:
        if col not in df_proc.columns: df_proc[col] = 0
    X = df_proc[MODEL_FEATURES].apply(pd.to_numeric, errors='coerce').fillna(0)
    dmatrix = xgb.DMatrix(X)
    return _model.get_booster().predict(dmatrix)

# 4. PRE-LOAD DATA
try:
    model, encoders, shap_values = load_assets()
    df = load_data()
    df['Probability'] = get_predictions(df, model, encoders)
    df['Is_Eligible'] = df['Probability'] > 0.5

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

def ui_card(title, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True)

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# 5. PAGE 1
def overview_page():
    st.markdown(f'<h1 style="color: #00327D;">Career Progression Platform {year} </h1>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Organisation-wide insights powered by Explainable AI.</div>',
                unsafe_allow_html=True)

    v_df = df.copy()

    # METRICS CARD
    def ui_card(title, value):
        st.markdown(
            f"""
            <div style="
                background-color: {MAIN_BLUE}; 
                padding: 30px 20px; 
                border-radius: 12px; 
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2); 
                text-align: center; 
                margin-bottom: 10px;
                border: 1px solid {MAIN_BLUE};">
                <div style="color: white; font-size: 14px; font-weight: normal; text-transform: uppercase; letter-spacing: 1px;">
                    {title}
                </div>
                <div style="color: white; font-size: 36px; font-weight: bold; margin-top: 10px;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    m1, m2, m3 = st.columns(3)
    with m1:
        ui_card("Total Employees", len(v_df))
    with m2:
        ui_card("Eligible For Promotion", int(v_df['Is_Eligible'].sum()))
    with m3:
        rate = (v_df['Is_Eligible'].mean() * 100) if not v_df.empty else 0
        ui_card("Average Promotion Rate", f"{rate:.1f}%")

    # CHART WITH SELECTION LOGIC
    with st.container(border=True):
        st.markdown(f'<h3 style="color: {MAIN_BLUE};">Eligible Employees by Department</h3>',
                    unsafe_allow_html=True)

        dept_stats = v_df.groupby('Department Name_x')['Is_Eligible'].sum().reset_index()
        dept_stats.columns = ['Department Name_x', 'Eligible_Count']
        dept_stats = dept_stats[dept_stats['Eligible_Count'] > 0]

        if not dept_stats.empty:
            fig = px.bar(
                dept_stats.sort_values('Eligible_Count', ascending=False),
                x='Department Name_x',
                y='Eligible_Count',
                labels={'Department Name_x': 'Department', 'Eligible_Count': 'Count of Eligible'},
                color_discrete_sequence=[LIGHT_BLUE]
            )

            fig.update_traces(texttemplate='%{y}', textposition='outside', cliponaxis=False,
                              textfont=dict(size=14, color=LIGHT_BLUE))
            fig.update_layout(
                barcornerradius=7,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Department Name",
                yaxis_title="Number of Eligible Employees",
                margin=dict(t=50),
                clickmode='event+select'
            )
            selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        else:
            st.write("No eligible employees found in any department.")
            selected_points = None

    # READY FOR PROMOTION TABLE (DYNAMICALLY FILTERED)
    st.markdown(f'<h3 style="color:{MAIN_BLUE};">Promotion Readiness</h3>', unsafe_allow_html=True)

    eligible_df = v_df[v_df['Is_Eligible'] == True].copy()

    if selected_points and "selection" in selected_points and selected_points["selection"]["points"]:
        clicked_dept = selected_points["selection"]["points"][0]["x"]
        eligible_df = eligible_df[eligible_df['Department Name_x'] == clicked_dept]
        st.caption(f"Showing eligible employees in **{clicked_dept}**")

    report_table = eligible_df[['Employee Name', 'Position Name', 'Probability']].copy()
    report_table['Status'] = report_table['Probability'].apply(
        lambda x: "Ready Now" if x >= 0.9 else "Ready with Development" if x >= 0.75 else "Not Ready")
    report_table['Probability'] = report_table['Probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(report_table, use_container_width=True, hide_index=True)

# 6. PAGE 2
def report_page():
    st.markdown(f'<h1 style="color: #00327D;">Employee Complete Report</h1>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Comprehensive promotion prediction and career development analysis.</div>',
                unsafe_allow_html=True)

    if 'reset_counter' not in st.session_state: st.session_state.reset_counter = 0

    def reset_report_filters():
        st.session_state.dept_rep = "All"
        st.session_state.sec_rep = "All"
        st.session_state.search_rep = ""
        st.session_state.reset_counter += 1

    #  SEARCH AND SELECTION
    st.markdown(f'<h3 style="color:{MAIN_BLUE};">Employee Search and Selection</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 2, 0.5])
    with c1:
        dept_opt = ["All"] + sorted(list(df['Department Name_x'].dropna().unique()))
        d_choice = st.selectbox("Department", dept_opt, key="dept_rep")
    with c2:
        sec_filtered = df[df['Department Name_x'] == d_choice][
            'Section Name'].dropna().unique() if d_choice != "All" else df['Section Name'].dropna().unique()
        s_choice = st.selectbox("Section", ["All"] + sorted(list(sec_filtered)), key="sec_rep")
    with c3:
        s_name = st.text_input("Search Name", placeholder="Type name..", key="search_rep")
    with c4:
        st.markdown("<div style='padding-top: 43px;'></div>", unsafe_allow_html=True)
        st.button("", icon=":material/refresh:", on_click=reset_report_filters, key="btn_rep_trigger")

    v_df_rep = df.copy()
    if d_choice != "All": v_df_rep = v_df_rep[v_df_rep['Department Name_x'] == d_choice]
    if s_choice != "All": v_df_rep = v_df_rep[v_df_rep['Section Name'] == s_choice]
    if s_name: v_df_rep = v_df_rep[v_df_rep['Employee Name'].str.contains(s_name, case=False, na=False)]

    selection = st.dataframe(
        v_df_rep[['Personnel Number', 'Employee Name', 'Position Name', 'Department Name_x']],
        column_config={
            "Personnel Number": "Staff ID",
            "Position Name": "Position",
            "Department Name_x": "Department"
        },
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        hide_index=True,
        key=f"table_{st.session_state.reset_counter}"
    )

    # EXPORT TO PDF
    def create_pdf(employee_data, p_info, n_info, met_count, total_reqs, prob, status_label):
        pdf = FPDF()
        pdf.add_page()

        # HEADER
        pdf.set_font("Arial", "B", 16)
        pdf.cell(190, 10, "Employee Career Progression Report", ln=True, align="C")
        pdf.set_font("Arial", "I", 8)
        pdf.cell(190, 5, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(10)

        # PROFILE SECTION
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(232, 238, 247)  # Light Blue Background
        pdf.cell(190, 10, " 1. Employee Profile", ln=True, fill=True)
        pdf.ln(2)

        pdf.set_font("Arial", "", 10)
        pdf.cell(100, 7, f"Name: {employee_data['Employee Name']}", ln=True)
        pdf.cell(100, 7, f"Staff ID: {employee_data['Personnel Number']}", ln=True)
        pdf.cell(100, 7, f"Position: {employee_data['Position Name']}", ln=True)
        pdf.cell(100, 7, f"Department: {employee_data['Department Name_x']}", ln=True)
        pdf.ln(5)

        # PROMOTION PREDICTION
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(232, 238, 247)
        pdf.cell(190, 10, " 2. Promotion Prediction", ln=True, fill=True)
        pdf.ln(2)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(100, 7, f"Promotion Likelihood: {prob:.1%}", ln=True)
        pdf.cell(100, 7, f"Mandatory Criteria Met: {met_count} / {total_reqs}", ln=True)

        # STATUS WITH DYNAMIC COLOURING
        if "NOT ELIGIBLE FOR PROMOTION" in status_label.upper():
            pdf.set_text_color(220, 53, 69)  # Red
        else:
            pdf.set_text_color(21, 87, 36)  # Green

        pdf.cell(100, 7, f"Status: {status_label.upper()}", ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset to Black
        pdf.ln(5)

        # PERFORMANCE HISTORY (PPA)
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(232, 238, 247)
        pdf.cell(190, 10, " 3. Performance History (PPA Ratings)", ln=True, fill=True)
        pdf.ln(2)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(30, 7, "Year", border=1, align="C")
        pdf.cell(40, 7, "Rating", border=1, align="C")
        pdf.ln()

        # Define the display mapping
        ppa_display_map = {
            "3.00": "3",
            "3.25": "3H",
            "3.50": "3S",
            "3.75": "3L"
        }

        pdf.set_font("Arial", "", 10)
        for yr in ['2021', '2022', '2023', '2024', '2025']:
            score_val = employee_data.get(yr, "N/A")

            # 🟢 Apply the Letter Grade logic for the PDF
            try:
                if score_val != "N/A":
                    score_key = f"{float(score_val):.2f}"
                    display_val = ppa_display_map.get(score_key, str(score_val))
                else:
                    display_val = "N/A"
            except:
                display_val = str(score_val)

            pdf.cell(30, 7, yr, border=1, align="C")
            pdf.cell(40, 7, display_val, border=1, align="C")
            pdf.ln()

        pdf.ln(5)

        # EXPLAINABLE AI (SHAP) INSIGHTS
        pdf.set_font("Arial", "B", 12)
        pdf.set_fill_color(232, 238, 247)
        pdf.cell(190, 10, " 4. AI Insights", ln=True, fill=True)
        pdf.ln(2)

        # STRENGTH
        pdf.set_font("Arial", "B", 10)
        pdf.cell(100, 7, f"Key Strength: {top_pos['Feature']}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, p_info['explain'])
        pdf.cell(100, 7, f"Recommended Action: {p_info['action']}", ln=True)
        pdf.ln(2)

        # DEVELOPMENT AREA
        pdf.set_font("Arial", "B", 10)
        pdf.cell(100, 7, f"Development Area: {top_neg['Feature']}", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, n_info['explain'])
        pdf.cell(100, 7, f"Recommended Action: {n_info['action']}", ln=True)
        pdf.ln(10)

        # SIGNATURE BLOCK
        pdf.set_font("Arial", "I", 8)
        pdf.cell(190, 10, "This report is generated by CareerUp and is intended for internal review only.",
                 ln=True, align="C")

        return pdf.output(dest='S').encode('latin-1')

    if selection.selection.rows:
        row_idx = selection.selection.rows[0]
        emp = v_df_rep.iloc[row_idx]
        st.divider()

        # PROFILE CARD
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                img_data = get_base64_image("person2.png")
                img_src = f"data:image/png;base64,{img_data}" if img_data else "https://www.w3schools.com/howto/img_avatar.png"
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <img src="{img_src}" style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover; margin-bottom: 15px;">
                        <h2 style="color: {LIGHT_BLUE}; margin: 0; font-size: 22px;">{emp['Employee Name']}</h2>
                        <p style="color: {OLIVE}; font-weight: normal; margin-top: 5px;">{emp['Personnel Number']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True)
            with col2:
                st.markdown(f'<h3 style="color:{MAIN_BLUE};">Employee Profile</h3>', unsafe_allow_html=True)

                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(f"**Email:** `emp_{emp['Personnel Number']}@petronas.com`")
                    st.markdown(f"**Position:** {emp['Position Name']}")
                    st.markdown(f"**Department:** {emp['Department Name_x']}")
                with p2:
                    st.markdown(f"**Salary Grade:** {emp['Salary Grade']}")
                    st.markdown(f"**Year in Salary Grade:** {emp['No of year in salary grade']:.1f} Years")
                    st.markdown(f"**Company:** Malaysian Refining Company Sdn. Bhd.")

        # st.markdown(f'<h3 style="color:{MAIN_BLUE};">Performance History (PPA)</h3>', unsafe_allow_html=True)
        # PERFORMANCE HISTORY (PPA) WITH TOOLTIPS
        st.markdown(
            f"""
            <style>
                .tooltip {{
                    position: relative;
                    display: inline-block;
                    cursor: help;
                    margin-left: 8px;
                }}
                .tooltip .tooltiptext {{
                    visibility: hidden;
                    width: 220px;
                    background-color: #A6A6A6;
                    color: #fff;
                    text-align: left;
                    border-radius: 6px;
                    padding: 10px;
                    position: absolute;
                    z-index: 100;
                    bottom: 125%; /* Position above the icon */
                    left: 50%;
                    margin-left: -110px;
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 14px;
                    font-weight: normal;
                    line-height: 1.5;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                }}
                .tooltip:hover .tooltiptext {{
                    visibility: visible;
                    opacity: 1;
                }}
                /* The little arrow under the tooltip */
                .tooltip .tooltiptext::after {{
                    content: "";
                    position: absolute;
                    top: 100%;
                    left: 50%;
                    margin-left: -5px;
                    border-width: 5px;
                    border-style: solid;
                    border-color: {GREY} transparent transparent transparent;
                }}
            </style>

            <h3 style="color:{MAIN_BLUE}; display: inline-block;">
                Performance History (PPA)
                <div class="tooltip">
                    <span style="color: #A6A6A6; font-size: 20px;">ⓘ</span>
                    <span class="tooltiptext">
                        <b>PPA Rating Scale</b><br>
                        1  - Exceptional<br>
                        2  - Very Strong<br>
                        3  - Meets Expectations<br>
                        3H - Minor Gaps<br>
                        3S  - Adequate<br>
                        3L - Below Expectations
                    </span>
                </div>
            </h3>
            """,
            unsafe_allow_html=True
        )

        # LABEL DEFINITION
        def get_ppa_label(score):
            try:
                s = float(score)
                if s <= 1.0: return "Exceptional", "#007A3F"
                if s <= 2.0: return "Very Strong", "#7EB14C"
                if s <= 3.0: return "Meets Expectations", "#9EB18B"
                if s <= 3.25: return "Minor Gaps", "#FFD230"
                if s <= 3.5: return "Adequate", "#F9B666"
                if s <= 3.75: return "Below Expectations", "#CA4A00"
                return "Unsatisfactory", "#BD0000"
            except:
                return "N/A", "#3D3E4A"

        # PERSONALISED CARDS WITH YEAR
        ppa_display_map = {
            "1.0" : "1",
            "2.0": "2",
            "3.0": "3",
            "3.25": "3H",
            "3.50": "3S",
            "3.75": "3L",
            "4.0": "4"
        }

        ppa_cols = st.columns(5)
        for i, year in enumerate(['2021', '2022', '2023', '2024', '2025']):
            score_val = emp.get(year, 0)
            label, color = get_ppa_label(score_val)

            try:
                # Convert to string with 2 decimal places to match the dictionary keys exactly
                score_key = f"{float(score_val):.2f}"

                # Strip the .00 if it's 3.00 to match "3.0"
                if score_key == "1.00": score_key = "1.0"
                if score_key == "2.00": score_key = "2.0"
                if score_key == "3.00": score_key = "3.0"
                if score_key == "4.00": score_key = "4.0"


                # Look up the letter (3H, 3S, etc). If not found, use the original number.
                display_val = ppa_display_map.get(score_key, score_val)
            except:
                display_val = score_val

            with ppa_cols[i]:
                # YEAR BADGE
                st.markdown(
                    f"""
                    <div style="
                        text-align: center; 
                        background-color: {LIGHT_BLUE}; 
                        color: white; 
                        font-size: 18px; 
                        font-weight: bold; 
                        padding: 5px 10px; 
                        border-radius: 10px; 
                        margin: 0 auto 12px auto;
                        box-shadow: 1px 1px 4px rgba(0,0,0,0.1);">
                        {year}
                    </div>
                    """,
                    unsafe_allow_html=True)

                # CARDS
                st.markdown(
                    f"""
                    <div style="background-color: white; padding: 15px 5px; border-radius: 12px; border: 1px solid #D7D7D8; text-align: center; min-height: 110px; display: flex; flex-direction: column; justify-content: center;">
                        <div style="color: {MAIN_BLUE}; font-size: 32px; font-weight: bold; margin-bottom: 5px;">{display_val}</div>
                        <div style="color: {color}; font-size: 11px; font-weight: normal; line-height: 1.2; padding: 0 5px;">
                            {label.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(f"<h3 </h3>", unsafe_allow_html=True)

        # ASSESSMENT AND COMPETENCY OVERVIEW
        with st.container(border=True):
            st.markdown(f'<h3 style= "color:{MAIN_BLUE};">Assessment and Competency Overview</h3>',
                        unsafe_allow_html=True)

            # DATA IN PERCENTAGE
            plot_data = pd.DataFrame({
                'Assessment': ['CCM Score', 'Edge', 'TECH'],
                'Score': [float(emp.get(f, 0)) * 100 for f in ['CCM Score (TSKG/BSKG)', 'Edge', 'TECH']]
            })

            fig_scores = px.bar(
                plot_data,
                x='Score',
                y='Assessment',
                orientation='h',
                text=plot_data['Score'].apply(lambda x: f'{x:.0f}%'),  # Explicitly format text here
                color_discrete_sequence=[LIGHT_BLUE]
            )

            fig_scores.update_traces(textposition='outside')
            fig_scores.update_layout(
                barcornerradius=7,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=250,
                xaxis=dict(range=[0, 115], showticklabels=False, showgrid=False),  # Hide axis to keep it clean
                yaxis_title=None,
                xaxis_title=None
            )

            st.plotly_chart(fig_scores, use_container_width=True)

            # ASSESSMENT CARDS
            b1, b2, b3, b4 = st.columns(4)

            with b1:
                st.info(f"**Health, Safety & Environment (HSE) Certification**\n\n"
                        f"{'Valid' if emp.get('Result HSE') == 2 else 'Expired' if emp.get('Result HSE') == 1 else 'Not Completed'}")

            with b2:
                st.info(f"**Technical Competency Assessment (PECAS)**\n\n"
                        f"{emp.get('PECAS EA / CAP MTG', 'N/A')}")

            with b3:
                # FIX: Ensure it checks 'COC Requirement PESITAS' for all conditions
                pesitas_val = emp.get('COC Requirement PESITAS')
                st.info(f"**Boiler & Engine Certification (PESITAS)**\n\n"
                        f"{'Complete' if pesitas_val == 2 else 'Incomplete' if pesitas_val == 1 else 'Exempted'}")

            with b4:
                # FIX: Ensure it consistently checks 'COC Requirement CHARGEMAN'
                chargeman_val = emp.get('COC Requirement CHARGEMAN')
                st.info(f"**Professional Chargeman Certification & Competency**\n\n"
                        f"{'Complete' if chargeman_val == 2 else 'Incomplete' if chargeman_val == 1 else 'Exempted'}")

        # CRITERIA LOGIC ENGINE
        skg = emp.get('Position SKG', '')
        cat = emp.get('PositionCategory', '')
        ppa = float(emp.get('PPA_Average', 0))

        checks = []
        total_reqs = 0

        # Logic 1: BSKG + Executive (4 Reqs)
        if skg == "BSKG" and cat == "E":
            checks = [
                emp.get('TECH', 0) > 0,
                emp.get('Edge', 0) > 0,
                emp.get('Result HSE') == 2,
                ppa <= 3.5
            ]
            total_reqs = 4
            req_names = ["TECH", "EDGE", "HSE", "PPA <= 3.5"]

        # Logic 2: BSKG + Non-Executive (2 Reqs)
        elif skg == "BSKG" and cat == "NE":
            checks = [
                emp.get('Result HSE') == 2,
                ppa <= 3.0
            ]
            total_reqs = 2
            req_names = ["HSE", "PPA <= 3.0"]

        # Logic 3: TSKG + Executive (5 Reqs)
        elif skg == "TSKG" and cat == "E":
            checks = [
                emp.get('TECH', 0) > 0,
                emp.get('Edge', 0) > 0,
                emp.get('CCM Score (TSKG/BSKG)', 0) > 0,
                emp.get('Result HSE') == 2,
                ppa <= 3.5
            ]
            total_reqs = 5
            req_names = ["TECH", "EDGE", "CCM", "HSE", "PPA <= 3.5"]

        # Logic 4: TSKG + Non-Executive (5 Reqs)
        elif skg == "TSKG" and cat == "NE":
            checks = [
                emp.get('Result HSE') == 2,
                emp.get('PECAS EA / CAP MTG') == 'Completed',
                emp.get('COC Requirement PESITAS') == 0,
                emp.get('COC Requirement CHARGEMAN') == 0,
                ppa <= 3.0
            ]
            total_reqs = 5
            req_names = ["HSE", "PECAS", "PESITAS", "CHARGEMAN", "PPA <= 3.0"]

        met_count = sum(checks)
        met_pct = (met_count / total_reqs) if total_reqs > 0 else 0

        # PROMOTION LIKELIHOOD
        st.markdown(f'<h3 style="color: {MAIN_BLUE};">Promotion Prediction</h3>', unsafe_allow_html=True)

        v1, v2 = st.columns([1, 1])
        CARD_HEIGHT = "220px"

        with v1:
            prob = float(emp.get('Probability', 0))
            prob_percent = prob * 100
            s_lbl, s_bg, s_txt = ("Eligible for Promotion", "#d4edda", "#155724") if prob > 0.5 else (
                "Not Eligible for Promotion", "#f8d7da", "#721c24")
            st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #f0f2f6; text-align: center; height: {CARD_HEIGHT}; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div style="color: {OLIVE}; font-size: 14px; font-weight: normal; text-transform: uppercase;">Promotion Likelihood</div>
                    <div style="color: {MAIN_BLUE}; font-size: 42px; font-weight: bold; margin-bottom: 5px;">{prob:.1%}</div>
                    <div style="background-color: #f0f2f6; border-radius: 10px; width: 100%; height: 10px; margin-bottom: 15px;">
                        <div style="background-color: {MAIN_BLUE}; width: {prob_percent}%; height: 100%; border-radius: 10px;"></div>
                    </div>
                    <div style="background-color: {s_bg}; color: {s_txt}; padding: 5px 15px; border-radius: 20px; font-size: 13px; font-weight: normal; border: 1px solid {s_txt}44;">{s_lbl}</div>
                </div>
            """, unsafe_allow_html=True)

        with v2:
            st.markdown(f"""
                <div style="background-color: white; padding: 25px; border-radius: 12px; border: 1px solid #f0f2f6; text-align: center; height: {CARD_HEIGHT}; display: flex; flex-direction: column; justify-content: center;">
                    <div style="color: {OLIVE}; font-size: 14px; font-weight: normal; text-transform: uppercase;">Criteria Met ({skg}-{cat})</div>
                    <div style="color: {MAIN_BLUE}; font-size: 48px; font-weight: bold; margin: 0;">{met_count}<span style="font-size: 20px; color: {OLIVE};"> / {total_reqs}</span></div>
                    <div style="color: {MAIN_BLUE}; font-size: 14px; font-weight: bold; margin-top: 5px;">Mandatory Requirements</div>
                </div>
            """, unsafe_allow_html=True)

        # ROADMAP MESSAGE
        missing_names = [name for i, name in enumerate(req_names) if not checks[i]]
        rec_msg = f"The candidate ({skg} {cat}) meets {met_count} out of {total_reqs} criteria ({met_pct:.0%}). "

        if met_pct == 1.0 and prob > 0.8:
            rec_msg += "All mandatory requirements are satisfied. The candidate is highly recommended for promotion review."
        else:
            focus = ", ".join(missing_names)
            rec_msg += f"To improve eligibility, focus on fulfilling the following: <b>{focus}</b>."

        # PROMOTION INSIGHTS
        st.markdown(f"""
            <div style="background-color: #fcf8e3; padding: 20px; border-radius: 12px; border: 1px solid #fcf8e3; margin-top: 20px; color: #8a6d3b;">
                <h4 style="margin-top:0; color: #8a6d3b;">Promotion Insights</h4>
                <p style="margin-bottom:0; font-size: 15px; line-height: 1.5;">{rec_msg}</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        #KEY FACTORS
        st.markdown(f'<h3 style= "color:{MAIN_BLUE};">Key Factors</h3>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-title">Feature importance showing how each factor contributes to the promotion prediction.</div>',
            unsafe_allow_html=True)

        try:
            target_id = emp['Personnel Number']
            global_idx = df[df['Personnel Number'] == target_id].index[0]
            shap_df = pd.DataFrame({'Feature': MODEL_FEATURES, 'Contribution': shap_values[global_idx]}).sort_values(
                'Contribution', ascending=False).head(5)

            fig_shap = px.bar(
                shap_df.sort_values('Contribution'),
                x='Contribution',
                y='Feature',
                orientation='h',
                color_discrete_sequence=['#7399C6'],
                text_auto='.2f'
            )

            fig_shap.update_traces(
                textfont=dict(
                    size=14,
                    color= LIGHT_BLUE
                ),
                textposition='outside',
                cliponaxis=False
            )

            max_val = shap_df['Contribution'].max()
            fig_shap.update_layout(
                barcornerradius=7,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[0, max_val * 1.25], showgrid=False)
            )

            fig_shap.update_traces(textposition='outside', cliponaxis=False)

            max_val = shap_df['Contribution'].max()
            min_val = shap_df['Contribution'].min()

            fig_shap.update_layout(
                barcornerradius=7,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(range=[min_val * 1.2 if min_val < 0 else 0, max_val * 1.2])
            )
            fig_shap.update_layout(barcornerradius=7, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

            with st.container(border=True):
                st.plotly_chart(fig_shap, use_container_width=True)
                # --- FULL SHAP EXPLANATION GRAPH (INSIDE EXPANDER) ---
                with st.expander("View Full AI Contribution Breakdown"):
                    st.markdown(f'<h4 style="color:{MAIN_BLUE};">SHAP Waterfall Analysis</h4>', unsafe_allow_html=True)
                    st.caption(
                        "This graph shows the 'Tug-of-War' between factors. Red pushes toward promotion, Blue pulls away.")

                    try:
                        # 1. Get the average probability as the starting point
                        base_value = df['Probability'].mean()

                        # 2. Extract values for this employee
                        if len(shap_values.shape) == 3:
                            current_sv = shap_values[1][global_idx]
                        else:
                            current_sv = shap_values[global_idx]

                        # 3. Create the Explanation Object
                        # Slicing to ensure lengths match available features
                        exp = shap.Explanation(
                            values=current_sv[:len(MODEL_FEATURES)],
                            base_values=base_value,
                            data=df[MODEL_FEATURES].iloc[global_idx].values,
                            feature_names=MODEL_FEATURES
                        )

                        # 4. Render the Waterfall Plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        # Use a transparent background to match Streamlit
                        fig.patch.set_alpha(0.0)

                        shap.plots.waterfall(exp, show=False)

                        # 5. Display in Streamlit
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()  # Always clear to save memory

                    except Exception as e:
                        st.warning(f"Detailed view unavailable for this staff. (Technical Error: {e})")

            top_pos, top_neg = shap_df.iloc[0], \
                pd.DataFrame({'Feature': MODEL_FEATURES, 'Contribution': shap_values[global_idx]}).sort_values(
                    'Contribution', ascending=True).iloc[0]

            # -CAREER COACHING LOGIC
            def get_shap_advice(feature_name, contribution):
                actual_val = emp.get(feature_name, 0)
                abs_cont = abs(contribution)

                # SHAP Intensity into HR Impact Levels
                if abs_cont > 1.0:
                    impact_lvl = "Significant"
                    priority_note = "This is the primary factor determining the current outcome."
                elif abs_cont > 0.5:
                    impact_lvl = "Moderate"
                    priority_note = "This area has a notable influence on promotion readiness."
                else:
                    impact_lvl = "Minimal"
                    priority_note = "This is a secondary factor to monitor during development."

                # TECH / TECH GAP
                if "TECH" in feature_name.upper():
                    if contribution < 0:
                        explain = f"Current technical competency levels show a gap compared to the target role requirements."
                        action = f"Prioritise technical upskilling to close the current gap of {abs(actual_val):.2f}."
                    else:
                        explain = f"Technical competency is a strong factor of career progression, significantly exceeding the requirements."
                        action = "Leverage this expertise by developing technical skills."

                # HSE
                elif "HSE" in feature_name.upper():
                    if actual_val < 2:
                        explain = f"Mandatory safety compliance requirements (HSE) are currently unmet."
                        action = "Complete the required HSE Certification to meet eligibility criteria."
                    else:
                        explain = "Full HSE compliance is maintained, satisfying a key organisational requirement."
                        action = "Continue to monitor expiry dates to ensure consistent eligibility."

                # PERFORMANCE RATINGS (PPA)
                elif any(yr in feature_name for yr in ['2021', '2022', '2023', '2024', '2025', 'PPA']):
                    if contribution < 0:
                        explain = f"Historical performance trends in {feature_name} are weighing down the overall readiness score."
                        action = "Focus on achieving an Exceptional or Very Strong rating in the current and following years to offset historical data."
                    else:
                        explain = f"Strong performance history in {feature_name} is providing a positive effect to the promotion profile."
                        action = "Maintain high-quality delivery to ensure this remains as a driving factor towards career progression."

                # EDGE
                elif "EDGE" in feature_name.upper():
                    if contribution < 0:
                        explain = f"EDGE competency is currently below the benchmark for the next level."
                        action = "Engage in leadership development programmes and actively seek guidance from supervisors."
                    else:
                        explain = f"Strong alignment with leadership behaviours is a key factor in the current evaluation."
                        action = "Apply these behavioural strengths in cross-functional task forces or department-wide projects."

                # CATCH-ALL
                else:
                    status = "Strength" if contribution > 0 else "Development Area"
                    explain = f"{feature_name} has been identified as a {impact_lvl} {status}."
                    action = "Discuss specific development steps with your line manager to optimise this factor."

                return {'explain': explain, 'action': action, 'impact': impact_lvl}

            p_info = get_shap_advice(top_pos['Feature'], top_pos['Contribution'])
            n_info = get_shap_advice(top_neg['Feature'], top_neg['Contribution'])

            st.markdown(f"""
                        <div style="background-color: #e8f4f8; padding: 25px; border-radius: 12px; border: 1px solid #d1e9f0;">
                            <h4 style="margin-top:0; color: #004085;">Insights</h4>
                            <div style="margin-top:15px;">
                                <p><b>Key Strength: {top_pos['Feature']}</b></p>
                                <p style="font-size: 14px; color: #444;">{p_info['explain']}</p>
                                <div style="background-color:#fff3cd; padding:10px; border-radius:8px; ">
                                    <b>Recommended Action:</b> {p_info['action']}
                                </div>
                            </div>
                            <hr>
                            <div>
                                <p><b>Development Area: {top_neg['Feature']}</b></p>
                                <p style="font-size: 14px; color: #444;">{n_info['explain']}</p>
                                <div style="background-color:#f8d7da; padding:10px; border-radius:8px; ">
                                    <b>Recommended Action:</b> {n_info['action']}
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        except:
            st.error("Error linking analysis.")

        # CAREER PATH RECOMMENDATIONS 1
        if emp['Employee Name'].strip().upper() == "UMMI BINTI MUHAMMAD IRFAN":
            st.divider()
            st.markdown(f'<h3 style="color: {MAIN_BLUE};">Career Path Recommendations</h3>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sub-title">Personalised career development roadmap and training recommendations.</div>',
                unsafe_allow_html=True)

            # TRAINING AND UPSKILLING RECOMMENDATIONS
            st.markdown(f'<h3 style="color: {MAIN_BLUE};">Training and Upskilling Recommendations</h3>',
                        unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)

            with t1:
                st.markdown(f"""
                    <div class="training-card">
                        <h4 style="color:{GREY}; margin-top:0;">Leadership Development</h4>
                        <p>Prepare for supervisory or senior executive responsibilities.</p>
                        <p><b>Recommended Training</b><br>
                        • Emerging Leaders Programme<br>
                        • Situational Leadership<br>
                        • Decision-Making and Accountability Workshops</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Improved confidence in leading initiatives<br>
                            • Enhanced strategic decision-making and accountability in complex environments
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with t2:
                st.markdown(f"""
                    <div class="training-card">
                        <h4 style="color:{GREY}; margin-top:0;">Strategic Business</h4>
                        <p>Understanding corporate management and long-term value creation.</p>
                        <p><b>Recommended Training</b><br>
                        • Strategic Planning Essentials<br>
                        • Financial Intelligence for Leaders<br>
                        • Change Management Frameworks</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Ability to align team goals with corporate strategy<br>
                            • Enhanced contribution to department-wide decision making
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with t3:
                st.markdown(f"""
                    <div class="training-card">
                        <h4 style="color:{GREY}; margin-top:0;">Stakeholder Management</h4>
                        <p>Master high-level corporate communication and negotiation.</p>
                        <p><b>Recommended Training</b><br>
                        • Executive Presence and Influence<br>
                        • Conflict Resolution for Managers<br>
                        • High-Stakes Negotiation Skills</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Stronger relationships with key stakeholders<br>
                            • Enhanced ability to align diverse interests and drive strategic initiatives
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.write("")

            # CERTIFICATIONS REQUIRED
            st.markdown(f'<h3 style="color: {LIGHT_BLUE};">Required Certifications for Career Advancement</h3>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            with c1:
                st.markdown(f"""
                <div class="cert-warning">
                    <h4 style="margin-top:0; color:{GREY};">HSE Certification</h4>
                    <p><b>Requirement:</b> Corporate HSE Compliance Module</p>
                    <p><b>Status:</b> Mandatory for next promotion level</p>
                    <p>Ummi must complete this module to meet baseline eligibility and resolve the primary bottleneck.</p>
                    <div class="outcome-box">
                        <b>Impact:</b> Full compliance with safety standards and removal of eligibility barriers.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="cert-card">
                    <h4 style="margin-top:0; color:{GREY};">Business Strategy Certificate</h4>
                    <p><b>Provider:</b> Coursera / edX (University-backed)</p>
                    <p><b>Duration:</b> 3-4 months of modular learning</p>
                    <p>If pursuing a pivot into strategic planning, this builds the necessary commercial thinking.</p>
                    <div class="outcome-box">
                        <b>Outcome:</b> Builds advanced strategic thinking required for future corporate leadership.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.write("")

            # ALTERNATIVE CAREER PATHS
            st.markdown(f'<h3 style="color: {GREY};">Alternative Career Paths</h3>', unsafe_allow_html=True)

            path_col1, path_col2 = st.columns(2)

            with path_col1:
                st.markdown(f"""
                    <div class="path-card">
                        <h4 style="color: {GREY}; margin-top:0;">Promotion Path</h4>
                        <p style="font-size: 18px; font-weight: normal; color: {GREY};">Senior Executive (Capability Management)</p>
                        <div style="font-size: 16px; line-height: 2;">
                            • Strong competency and skill alignment<br>
                            • Meets all performance and assessment thresholds<br>
                            • Developmental gaps are manageable and non-structural
                        </div>
                        <p style="font-size: 14px; color: {OLIVE}; margin-top: 20px; font-style: italic;">
                            Recommended trajectory based on current technical mastery and leadership potential.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with path_col2:
                st.markdown(f"""
                    <div class="path-card">
                        <h4 style="color:{GREY}; margin-top:0;">Lateral Paths</h4>
                        <p style="font-size: 18px; font-weight: normal; color: {GREY};">Cross-Functional Opportunities</p>
                        <div style="font-size: 16px; line-height: 2;">
                            • Talent Development Executive<br>
                            • Organisational Development Executive<br>
                            • HR Business Partner
                        </div>
                        <p style="font-size: 14px; color: {OLIVE}; margin-top: 20px; font-style: italic;">
                            Paths identified based on competency overlap and corporate management minor.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.write("")

            # MILESTONES & ACTION PLAN
            st.markdown(f'<h3 style="color: {MAIN_BLUE};">Career Development Milestones and Action Plan</h3>',
                        unsafe_allow_html=True)
            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{MAIN_BLUE};'>Short-Term (0–6 months)</h4>", unsafe_allow_html=True)
                    st.checkbox("Complete leadership fundamentals training", key="ms_short_1")
                    st.checkbox("Participate in cross-department initiative", key="ms_short_2")
                    st.checkbox("Improve analytical skills", key="ms_short_3")
                    st.markdown('</div>', unsafe_allow_html=True)

            with m_col2:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{LIGHT_BLUE};'>Mid-Term (6–12 months)</h4>", unsafe_allow_html=True)
                    st.checkbox("Obtain at least one professional certification", key="ms_mid_1")
                    st.checkbox("Lead a capability upskilling project", key="ms_mid_2")
                    st.checkbox("Measurable impact on competency metrics", key="ms_mid_3")
                    st.markdown('</div>', unsafe_allow_html=True)

            with m_col3:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{GREY};'>Long-Term (12–18 months)</h4>", unsafe_allow_html=True)
                    st.checkbox("Act as project lead in strategic initiatives", key="ms_long_1")
                    st.checkbox("Achieve consistent PPA", key="ms_long_2")
                    st.checkbox("Ready for promotion review", key="ms_long_3")
                    st.markdown('</div>', unsafe_allow_html=True)

        # CAREER PATH RECOMMENDATIONS 2
        if emp['Employee Name'].strip().upper() == "MUHAMMAD IQBAL BIN NAUFAL":
            st.divider()
            st.markdown(f'<h3 style="color: {MAIN_BLUE};">Career Path Recommendations</h3>', unsafe_allow_html=True)
            st.markdown(
                '<div class="sub-title">Personalised career development roadmap and training recommendations.</div>',
                unsafe_allow_html=True)

            # TRAINING
            st.markdown(f'<h3 style="color: {MAIN_BLUE};">Training and Upskilling Recommendations</h3>',
                        unsafe_allow_html=True)

            st.error(
                "Current non-eligibility is driven by a technical skill gap (–1.0 TECH gap) and a need for stronger analytical maturity.")

            t1, t2, t3 = st.columns(3)
            with t1:
                st.markdown(f"""
                    <div class="training-card">
                        <h4 style="color:{GREY}; margin-top:0;">Core Technical Remediation</h4>
                        <p><b>Goal:</b> Close the –1.0 TECH gap in RBI & Assurance</p>
                        <p><b>Recommended Training</b><br>
                        • RBI Framework Deep Dive<br>
                        • Risk-Based Auditing in Practice<br>
                        • Internal Control Design and Testing</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Improved technical judgement<br>
                            • Higher-quality assurance outputs
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with t2:
                st.markdown(f"""
                    <div class="training-card" style="border-left-color: {OLIVE};">
                        <h4 style="color:{GREY}; margin-top:0;">Analytical and Digital Skills</h4>
                        <p><b>Goal:</b> Reduce reliance on manual and template-based work</p>
                        <p><b>Recommended Training</b><br>
                        • Data Analytics for Audit and Assurance<br>
                        • Excel / Power BI for Risk Analysis<br>
                        • Audit Automation Tools</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Stronger analytical evidence<br>
                            • Measurable TECH score improvement
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with t3:
                st.markdown(f"""
                    <div class="training-card" style="border-left-color: {MAIN_BLUE};">
                        <h4 style="color:{GREY}; margin-top:0;">Technical Problem-Solving</h4>
                        <p><b>Goal:</b> Improve independent execution and quality</p>
                        <p><b>Recommended Training</b><br>
                        • Root Cause Analysis (RCA)<br>
                        • Assurance Findings<br>
                        • Quality Assurance Methodologies</p>
                        <div class="outcome-box">
                            <b>Expected Outcome</b><br>
                            • Higher-impact audit findings<br>
                            • Improved confidence in technical reviews
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.write("")

            # CERTIFICATIONS
            st.markdown(f'<h3 style="color: {LIGHT_BLUE};">Required Certifications for Promotion Eligibility</h3>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                    <div class="cert-warning">
                        <h4 style="margin-top:0; color:{GREY};">HSE Certification</h4>
                        <p><b>Requirement:</b> Corporate HSE Compliance Module</p>
                        <p><b>Status:</b> Mandatory for next promotion level</p>
                        <p>Muhammad Iqbal must complete this module to meet baseline eligibility and resolve the primary bottleneck.</p>
                        <div class="outcome-box">
                            <b>Impact:</b> Full compliance with safety standards and removal of eligibility barriers.
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                    <div class="cert-card">
                        <h4 style="margin-top:0; color:{GREY};">Internal Audit and Risk Certification</h4>
                        <p><b>Target:</b> Certified Internal Auditor (CIA) or relevant technical certification.</p>
                        <p><b>Status:</b> Professional validation requirement</p>
                        <p>Obtain professional certification to validate technical core and bridge the gap between behavioural performance and technical depth.</p>
                        <div class="outcome-box">
                            <b>Impact:</b> Strengthened technical credibility and alignment with role competency standards.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.write("")
            st.write("")

            # ALTERNATIVE CAREER PATHS
            st.markdown(f"<h3>Alternative Career Paths</h3>", unsafe_allow_html=True)
            path_col1, path_col2, path_col3 = st.columns(3)

            with path_col1:
                st.markdown(f"""
                    <div class="path-card">
                        <h4 style="color:{GREY}; margin-top:0;">A. Technical Track</h4>
                        <p><b>Focus:</b> Remain in RBI and Assurance</p>
                        <p>Continue deep specialisation to reach the minimum TECH threshold for Senior Executive roles.</p>
                    </div>
                """, unsafe_allow_html=True)

            with path_col2:
                st.markdown(f"""
                    <div class="path-card">
                        <h4 style="color:{GREY}; margin-top:0;">B. Lateral Tech Stab.</h4>
                        <p><b>Focus:</b> Process Support</p>
                        <p>Suitable if deep technical auditing specialisation remains a challenge despite remediation.</p>
                    </div>
                """, unsafe_allow_html=True)

            with path_col3:
                st.markdown(f"""
                    <div class="path-card">
                        <h4 style="color: {GREY}; margin-top:0;">C. Non-Tech Transition</h4>
                        <p><b>Focus:</b> Operational Roles</p>
                        <p>Only recommended if the TECH gap persists after the 12-month intervention period.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write("")

            # MILESTONES & ACTION PLAN
            st.markdown(f'<h3 style="color: {GREY};">Career Development Milestones and Action Plan</h3>',
                        unsafe_allow_html=True)

            m_col1, m_col2, m_col3 = st.columns(3)

            with m_col1:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{MAIN_BLUE};'>Short-Term (0–6 months)</h4>", unsafe_allow_html=True)
                    st.checkbox("Enrol in core RBI and technical training", key="has_short_1")
                    st.checkbox("Improve audit documentation quality", key="has_short_2")
                    st.checkbox("Reduce rework through improved compliance", key="has_short_3")
                    st.markdown('</div>', unsafe_allow_html=True)

            with m_col2:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{LIGHT_BLUE};'>Mid-Term (6–12 months)</h4>",
                                unsafe_allow_html=True)
                    st.checkbox("Complete one professional certification", key="has_mid_1")
                    st.checkbox("Demonstrate improved analytical skills", key="has_mid_2")
                    st.checkbox("Improve CCM score", key="has_mid_3")
                    st.markdown('</div>', unsafe_allow_html=True)

            with m_col3:
                with st.container(border=True):
                    st.markdown(f"<h4 style='color:{GREY};'>Long-Term (12–24 months)</h4>", unsafe_allow_html=True)
                    st.checkbox("Achieve TECH score above threshold", key="has_long_1")
                    st.checkbox("Improve EDGE competency", key="has_long_2")
                    st.checkbox("Re-evaluation for promotion eligibility", key="has_long_3")
                    st.markdown('</div>', unsafe_allow_html=True)

        # EXPORT TO PDF
        st.write("")
        st.write("")
        try:
            pdf_bytes = create_pdf(
                employee_data=emp,
                p_info=p_info,
                n_info=n_info,
                met_count=met_count,
                total_reqs=total_reqs,
                prob=prob,
                status_label=s_lbl
            )

            st.download_button(
                label="Download Full Report as PDF",
                data=pdf_bytes,
                file_name=f"Report_{emp['Personnel Number']}.pdf",
                mime="application/pdf",
                icon=":material/download:",
                use_container_width=True
            )
        except:
            st.error(f"Error generating PDF")
    else:
        st.info("Please select an employee from the table above.")


# 6. NAVIGATION
pg_overview = st.Page(overview_page, title="Overview", default=True)
pg_report = st.Page(report_page, title="Employee Complete Report")
nav = st.navigation([pg_overview, pg_report], position="sidebar")
try:
    st.logo("careerup2.png", size="large")
except:
    with st.sidebar:
        st.write("### CAREER UP")
nav.run()


