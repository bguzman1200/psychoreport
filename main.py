import streamlit as st
import pandas as pd
import re
from fuzzywuzzy import process, fuzz
import plotly.express as px
from openai import OpenAI
from docx import Document
from io import BytesIO
import random

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="PsychoReport.ai",
    layout="wide",
    page_icon="🧠"
)

# Initialize Session State for API Key if not present
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""

# --- 2. DOMAIN KNOWLEDGE BASE ---
# Keywords for Fuzzy Matching
DOMAINS = {
    "Inhibition": ["interrupt", "blurt", "turn", "stop", "calm", "yell", "impulsive", "talk"],
    "Shift/Flexibility": ["stuck", "transition", "change", "new", "routine", "flexible", "adapt"],
    "Organization/Plan": ["messy", "lose", "backpack", "materials", "time", "due", "plan", "forget"]
}

# Likert Scale Mapping
LIKERT_MAP = {
    "never": 1, "rarely": 1, "1": 1,
    "sometimes": 2, "2": 2,
    "often": 3, "3": 3,
    "very often": 4, "very": 4, "4": 4,
    "almost always": 4, "5": 4 # Handling potential 5-point scales by capping or mapping high
}

# --- 3. HELPER FUNCTIONS ---

def clean_header(header_text):
    """
    Extracts the core behavior from a Google Form header.
    Priority: Text inside brackets [], otherwise last 5 words.
    """
    # Regex to find text inside brackets
    match = re.search(r'\[(.*?)\]', str(header_text))
    if match:
        return match.group(1).strip()
    
    # Fallback: Take last 5 words
    words = str(header_text).split()
    return " ".join(words[-5:])

def map_domain(cleaned_header):
    """
    Maps a cleaned header string to a Domain using Fuzzy Matching.
    """
    best_score = 0
    best_domain = "Uncategorized"
    
    for domain, keywords in DOMAINS.items():
        # Compare header against all keywords in this domain
        # extractOne returns (match, score)
        match, score = process.extractOne(cleaned_header, keywords, scorer=fuzz.partial_token_sort_ratio)
        
        # We accumulate confidence. If a keyword matches strongly (>80), likely that domain.
        if score > best_score:
            best_score = score
            best_domain = domain
            
    # Threshold: If score is too low, keep Uncategorized (Optional, currently greedy)
    return best_domain if best_score > 60 else "Uncategorized"

def score_value(val):
    """Converts string/int input to 1-4 integer score."""
    if pd.isna(val):
        return 1
    
    s_val = str(val).lower().strip()
    
    # Direct fuzzy lookup in map could be safer, but direct dict check is fast
    for key, score in LIKERT_MAP.items():
        if key in s_val:
            return score
            
    # Fallback try to extract first digit
    try:
        return int(re.search(r'\d+', s_val).group())
    except:
        return 1

def process_csv(uploaded_file):
    """
    The 'Brain': Cleans, Maps, and Scores the uploaded CSV.
    Returns: Dict of Domain Scores, Dict of Comments
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

    scores = {d: [] for d in DOMAINS.keys()}
    comments = []

    # Identify Question Columns vs Timestamp/Score
    for col in df.columns:
        # Skip metadata columns
        if "timestamp" in col.lower() or "score" in col.lower() or "email" in col.lower() or "name" in col.lower():
            continue
            
        # Check if it's a comment box
        if "comment" in col.lower() or "note" in col.lower():
            # Gather all non-empty comments
            valid_comments = df[col].dropna().tolist()
            comments.extend([str(c) for c in valid_comments if str(c).strip()])
            continue

        # Processing Rating Columns
        cleaned = clean_header(col)
        domain = map_domain(cleaned)
        
        if domain in scores:
            # Convert column to numeric
            numeric_col = df[col].apply(score_value)
            # Add mean of this column to the domain list (or all values?)
            # Prompt implies we need an aggregate score for the domain.
            # Let's take the mean of the column (average frequency for that specific behavior across all rows if multiple entries, 
            # but usually these are single student reports. Assuming 1 row per file for 'This Student')
            
            # If multiple rows (e.g. multiple teachers), we average them.
            col_mean = numeric_col.mean()
            scores[domain].append(col_mean)

    # Calculate Final Domain Means
    final_scores = {}
    for domain, values in scores.items():
        if values:
            final_scores[domain] = round(sum(values) / len(values), 2)
        else:
            final_scores[domain] = 0.0

    return final_scores, "\n".join(comments)

def generate_dummy_csv():
    """Generates realistic dummy data for testing."""
    data = {
        "Timestamp": ["2023-10-27 10:00:00"],
        "Student Name": ["Alex Doe"],
        "How often does the student [Interrupts others]?": [random.choice(["Often", "Very Often"])],
        "Rate frequency [Blurts out answers]": [random.choice(["Sometimes", "Often"])],
        "Frequency [Difficulty transitioning]": [random.choice(["Often", "Very Often"])],
        "Observe [Gets stuck on topics]": [random.choice(["Sometimes", "Rarely"])],
        "Rate [Messy backpack]": [random.choice(["Very Often", "Often"])],
        "Rate [Loses materials]": [4],
        "Additional Comments": ["Struggles significantly with impulse control during math."]
    }
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key Input
    api_key_input = st.text_input("OpenAI API Key", type="password", help="Required for report generation.")
    if api_key_input:
        st.session_state["api_key"] = api_key_input
        
    report_tone = st.selectbox("Report Tone", ["Clinical", "Parent-Friendly", "IEP-Focused"])
    
    st.info("🔒 **Privacy Mode**: Data is processed in-memory and wiped on refresh. No data is saved to any server.")
    
    st.divider()
    
    # Generate Sample Data
    st.write("**New? Test with Sample Data:**")
    if st.button("Generate Sample CSVs"):
        csv_data = generate_dummy_csv()
        st.download_button("Download Teacher Sample", csv_data, "teacher_sample.csv", "text/csv")
        st.download_button("Download Parent Sample", csv_data, "parent_sample.csv", "text/csv")
        st.download_button("Download Observer Sample", csv_data, "observer_sample.csv", "text/csv")

# --- 5. MAIN INTERFACE ---
st.title("🧠 PsychoReport.ai")
st.markdown("### Executive Functioning Report Generator")
st.markdown("Upload Google Form CSV exports to analyze Inhibition, Shift, and Organization domains.")

col1, col2, col3 = st.columns(3)
with col1:
    teacher_file = st.file_uploader("Teacher Rating (CSV)", type=["csv"])
with col2:
    parent_file = st.file_uploader("Parent Rating (CSV)", type=["csv"])
with col3:
    observer_file = st.file_uploader("Observer Rating (CSV)", type=["csv"])

# --- 6. DATA PROCESSING & VISUALIZATION ---
if teacher_file or parent_file or observer_file:
    st.divider()
    st.subheader("📊 Domain Analysis")
    
    all_scores = []
    
    # Process Files
    raters = {
        "Teacher": teacher_file,
        "Parent": parent_file, 
        "Observer": observer_file
    }
    
    aggregated_results = {} # Store for AI

    for rater_name, file in raters.items():
        if file:
            scores, comments = process_csv(file)
            if scores:
                aggregated_results[rater_name] = scores
                # Prepare data for Plotly
                for domain, score in scores.items():
                    all_scores.append({"Rater": rater_name, "Domain": domain, "Score": score})

    # Visualization
    if all_scores:
        df_viz = pd.DataFrame(all_scores)
        
        # Grouped Bar Chart
        fig = px.bar(
            df_viz, 
            x="Domain", 
            y="Score", 
            color="Rater", 
            barmode="group",
            range_y=[0, 4.5],
            title="Cross-Informant Domain Comparison"
        )
        
        # Clinical Significance Line
        fig.add_hline(y=3.0, line_dash="dot", line_color="red", annotation_text="Clinical Significance (3.0)")
        
        st.plotly_chart(fig, use_container_width=True)

    # --- 7. AI REPORT GENERATOR ---
    st.divider()
    col_gen, col_download = st.columns([1, 1])
    
    with col_gen:
        generate_btn = st.button("📝 Generate Clinical Report", type="primary")

    if generate_btn:
        if not st.session_state["api_key"]:
            st.error("Please enter your OpenAI API Key in the sidebar.")
        else:
            with st.spinner("Triangulating data and writing report..."):
                try:
                    client = OpenAI(api_key=st.session_state["api_key"])
                    
                    # Construct Prompt
                    prompt_text = f"""
                    You are a Licensed School Psychologist writing a formal psycho-educational report.
                    Tone: {report_tone}.
                    
                    TASK: Write a triangulated Executive Functioning analysis based on the following data (Scale 1-4, >3.0 is clinically significant):
                    {aggregated_results}
                    
                    STRUCTURE:
                    1. Cross-Informant Analysis: Compare raters (Home vs School). Highlight discrepancies > 1.0 points.
                    2. Domain Breakdown:
                       - Inhibition
                       - Shift/Flexibility
                       - Organization/Planning
                       (Label scores > 3.0 as "Clinically Significant Weakness")
                    3. Recommendations: Provide 3 specific, classroom-based accommodations for the weakest area.
                    
                    Do not include a header or footer in the text, just the body content.
                    """

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful expert School Psychologist assistant."},
                            {"role": "user", "content": prompt_text}
                        ],
                        stream=True
                    )
                    
                    # Stream output
                    report_container = st.empty()
                    full_report = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_report += content
                            report_container.markdown(full_report)
                            
                    st.session_state["generated_report"] = full_report
                    st.success("Report Generated!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # --- 8. WORD DOC EXPORT ---
    if "generated_report" in st.session_state:
        # Create Docx
        doc = Document()
        doc.add_heading("Confidential Psycho-Educational Report", 0)
        doc.add_paragraph(st.session_state["generated_report"])
        
        footer = doc.sections[0].footer
        p = footer.paragraphs[0]
        p.text = "Generated by PsychoReport.ai. Draft only."
        
        # Save to buffer
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        
        with col_download:
            st.download_button(
                label="📄 Download Word Doc",
                data=buffer,
                file_name="PsychoReport_Draft.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
