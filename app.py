# app.py
from __future__ import annotations

# IMPORTANT: st.set_page_config() must be the FIRST Streamlit command
import streamlit as st
st.set_page_config(page_title="SmartVEGFR", layout="wide", page_icon="LOGO.jpeg")

import base64
from io import BytesIO
from pathlib import Path
import os
import warnings

import base64

from math import ceil
from PIL import Image     


# Suppress some warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import joblib
import numpy as np
import pandas as pd

from fpdf import FPDF  # Use fpdf2 library
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski, MolToSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import rdMolDraw2D

# Tensorflow for Keras model
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
     TF_AVAILABLE = False
     st.warning("TensorFlow not found. Cannot load Keras (.h5) models.")


# --- Matplotlib Style ---
sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


################################################################################
# Helper utilities & Asset Loading
################################################################################

# CRITICAL: Use the same parameters as training
FP_RADIUS = 3
FP_BITS = 2048
MODEL_DIR = "saved_model" # Directory where model and selector are saved

@st.cache_resource(show_spinner="Loading model and features...")
def load_assets():
    """Load the persisted model + selector, handling different file types."""
    model = None
    selector = None
    model_type = "Unknown"
    
    model_dir_path = Path(MODEL_DIR)
    selector_path = model_dir_path / "feature_selector.pkl"
    model_path_joblib = model_dir_path / "best_model.joblib"
    model_path_pkl = model_dir_path / "best_model.pkl"
    model_path_h5 = model_dir_path / "best_model.h5"

    if not model_dir_path.exists() or not selector_path.exists():
         st.error(
            f"❌ Error: Files not found. Make sure the '{MODEL_DIR}' folder exists with "
            f"'feature_selector.pkl' and model file ('best_model.joblib' or 'best_model.h5') alongside app.py"
        )
         st.stop()
         
    selector = joblib.load(selector_path)

    if model_path_h5.exists() and TF_AVAILABLE:
        model = load_model(model_path_h5)
        model_type = "Keras/FCNN"
    elif model_path_joblib.exists():
         model = joblib.load(model_path_joblib)
         model_type = "Scikit-learn (joblib)"
    elif model_path_pkl.exists(): # Fallback for .pkl
         model = joblib.load(model_path_pkl) # joblib can load pkl
         model_type = "Scikit-learn (pkl)"
    else:
        st.error(f"❌ Error: No model file found (joblib, pkl, h5) in '{MODEL_DIR}' folder.")
        st.stop()
        
    return model, selector, model_type

# Load assets immediately
try:
   MODEL, SELECTOR, MODEL_TYPE_NAME = load_assets()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

def calculate_properties(mol):
    """Calculate ADME/Physicochemical properties."""
    if mol is None:
         return pd.Series({ 'MW': np.nan, 'LogP': np.nan,'HBD': np.nan,'HBA': np.nan, 'PSA': np.nan, 'Rotatable_Bonds': np.nan,'Drug_Like': False, 'Violates_RO5': True })
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        psa = Descriptors.TPSA(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        ro5_violations = (mw > 500 or logp > 5 or hbd > 5 or hba > 10)
        drug_like = not ro5_violations
        return pd.Series({
            'MW': mw, 'LogP': logp, 'HBD': hbd,'HBA': hba, 'PSA': psa,
             'Rotatable_Bonds': rotatable_bonds,'Drug_Like': drug_like, 'Violates_RO5': ro5_violations
        })
    except:
         return pd.Series({ 'MW': np.nan, 'LogP': np.nan,'HBD': np.nan,'HBA': np.nan, 'PSA': np.nan, 'Rotatable_Bonds': np.nan,'Drug_Like': False, 'Violates_RO5': True })

@st.cache_data()
def process_smiles(smiles_list):
    """Convert list of SMILES to mols, fingerprints, and properties."""
    mols, fps, valid_smiles, properties_list = [], [], [], []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                 # Fingerprint generation
                 fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_BITS)
                 fp_array = np.array(list(fp))
                 mols.append(mol)
                 fps.append(fp_array)
                 valid_smiles.append(smi)
                 properties_list.append(calculate_properties(mol))
            else:
                 st.warning(f"Invalid SMILES skipped: {smi}")
        except Exception as e:
             st.warning(f"Error processing {smi}: {e}")
             
    if not fps:
        return None, None, None, None
        
    properties_df = pd.DataFrame(properties_list)
    return np.array(fps), mols, valid_smiles, properties_df

def fig_to_buf(fig):
    """Convert Matplotlib figure to BytesIO buffer."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig) # CRITICAL: Close figure to free memory
    buf.seek(0)
    return buf
    
def mol_grid_to_buf(mols, legends, molsPerRow=4):
    """Convert RDKit Mol Grid Image to BytesIO buffer."""
    try:
       img = Draw.MolsToGridImage(
            mols,
            molsPerRow=molsPerRow,
            subImgSize=(250, 250),
            legends=legends,
            returnPNG=False # Returns PIL Image
        )
       buf = BytesIO()
       img.save(buf, format="PNG")
       buf.seek(0)
       return buf
    except Exception as e:
        st.warning(f"Failed to create molecular grid: {e}")
        return None

# Define a custom PDF class to add header/footer
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI-Driven Virtual Screening Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        try:
            # For newer fpdf versions with page numbering
            self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
        except:
            # Fallback for older versions
            self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, title):
         self.add_page()
         self.set_font('Arial', 'B', 14)
         self.cell(0, 10, title, 0, 1, 'L')
         self.ln(4)

    def add_image_centered(self, buf, img_width_pt=450):
         page_width = self.w - self.l_margin - self.r_margin
         x_centered = (page_width - img_width_pt) / 2 + self.l_margin
         # Ensure image fits, adjust width if needed, but keep aspect ratio
         self.image(buf, x=x_centered,  w=img_width_pt)
         self.ln(10) # Add space after image

################################################################################
# Visualisations & PDF Building
################################################################################

def create_all_visualisations(df: pd.DataFrame, fps_array: np.ndarray):
    """Return a list of (title, BytesIO PNG buffer) for inclusion in PDF and display."""
    images = []
    n_compounds = len(df)
    
    # 1. Bar Plot of Activity
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Compound', y='Activity (%)', data=df, palette='viridis', ax=ax1)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f'Predicted Activity of Top {n_compounds} Compounds')
    plt.xlabel('Compound')
    plt.ylabel('Predicted Activity (%)')
    plt.tight_layout()
    images.append((f"1. Predicted Activity - Top {n_compounds}", fig_to_buf(fig1)))

    # 2. Chemical Space (MW vs LogP)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    scatter = sns.scatterplot(data=df, x='MW', y='LogP', hue='Activity (%)', size='Activity (%)', 
                             palette='viridis', sizes=(50, 300), ax=ax2, legend='brief')
    plt.title(f'Chemical Space (MW vs LogP) - Top {n_compounds}')
    plt.xlabel('Molecular Weight')
    plt.ylabel('LogP')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    images.append((f"2. Chemical Space (MW vs LogP) - Top {n_compounds}", fig_to_buf(fig2)))

    # 3. Histograms of Properties
    fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    props = ['MW', 'LogP', 'PSA', 'HBD', 'HBA', 'Rotatable_Bonds']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, prop in enumerate(props):
       sns.histplot(df[prop], bins=min(10, n_compounds), kde=True, color=colors[i], alpha=0.6, ax=axes[i])
       axes[i].set_title(prop)
       axes[i].set_xlabel('')
    plt.tight_layout()
    fig3.suptitle(f'Distribution of Properties - Top {n_compounds}', y=1.02, fontsize=14)
    images.append((f"3. Property Distributions - Top {n_compounds}", fig_to_buf(fig3)))

    # 4. t-SNE Chemical Space (handle perplexity for small N)
    if n_compounds > 5: # t-SNE needs enough points
        try:
             # Perplexity must be less than n_samples
             perplexity_val = min(15, n_compounds - 1) 
             if perplexity_val > 0:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, learning_rate='auto', init='pca')
                # Use raw fingerprints for t-SNE, not selected features
                embedding = tsne.fit_transform(fps_array) 
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                scatter = ax4.scatter(embedding[:, 0], embedding[:, 1], c=df['Activity (%)'], cmap='viridis', alpha=0.8)
                plt.colorbar(scatter, label='Predicted Activity (%)')
                
                # Highlight top 5 compounds if we have enough
                if n_compounds >= 10:
                    top_n_to_highlight = min(5, n_compounds)
                    top_n_indices = df.head(top_n_to_highlight).index
                    ax4.scatter(embedding[df.index.isin(top_n_indices), 0],
                                embedding[df.index.isin(top_n_indices), 1],
                                facecolors='none', edgecolors='red', s=200, 
                                linewidth=2, label=f'Top {top_n_to_highlight} Hits')
                    ax4.legend()
                
                plt.title(f'Chemical Space (t-SNE) - Top {n_compounds}')
                plt.xlabel("t-SNE Dimension 1")
                plt.ylabel("t-SNE Dimension 2")
                plt.tight_layout()
                images.append((f"4. Chemical Space (t-SNE) - Top {n_compounds}", fig_to_buf(fig4)))
        except Exception as e:
             st.warning(f"Could not generate t-SNE plot (need more compounds or adjust perplexity): {e}")
             
    # 5. Molecule Grid Image
    mols = df['mol'].tolist()
    legends = [f"{row['Compound']} ({row['Activity (%)']:.1f}%)" for index, row in df.iterrows()]
    mols_per_row = 5 if n_compounds > 15 else (4 if n_compounds > 8 else 3)

    # عدد الصور المطلوبة
    chunk_size = 40
    total_chunks = ceil(len(mols) / chunk_size)

    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        mols_chunk = mols[start:end]
        legends_chunk = legends[start:end]

        mol_grid_buf = mol_grid_to_buf(mols_chunk, legends_chunk, molsPerRow=mols_per_row)
        
        if mol_grid_buf:
            images.append((f"5. Structures - Compounds {start + 1} to {min(end, len(mols))}", mol_grid_buf))

    # 6. Violin plots for key properties by activity (binarize activity for visualization)
    if n_compounds > 5:
        # Create a binarized activity column for visualization
        df_viz = df.copy()
        median_activity = df_viz['Activity (%)'].median()
        df_viz['Activity_Group'] = df_viz['Activity (%)'].apply(lambda x: 'High' if x >= median_activity else 'Low')
        
        # Violin plot for Molecular Weight
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        sns.violinplot(x='Activity_Group', y='MW', data=df_viz, hue='Activity_Group', 
                      split=True, legend=False, palette='viridis')
        plt.title('Molecular Weight Distribution by Activity Group', fontsize=14)
        plt.xticks([0, 1], ['Low Activity', 'High Activity'], rotation=0)
        plt.ylabel('Molecular Weight', fontsize=12)
        plt.xlabel('Activity Group', fontsize=12)
        plt.tight_layout()
        images.append((f"6. MW Distribution by Activity - Top {n_compounds}", fig_to_buf(fig6)))

        # Box plot for LogP
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Activity_Group', y='LogP', data=df_viz, hue='Activity_Group', 
                   palette='Set2', legend=False)
        plt.title('LogP Distribution by Activity Group', fontsize=14)
        plt.xticks([0, 1], ['Low Activity', 'High Activity'], rotation=0)
        plt.ylabel('LogP', fontsize=12)
        plt.xlabel('Activity Group', fontsize=12)
        plt.tight_layout()
        images.append((f"7. LogP Distribution by Activity - Top {n_compounds}", fig_to_buf(fig7)))

    # 7. Pair plot for top compounds (only if reasonable number)
    
    # 8. Correlation heatmap
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    corr = df[['MW', 'LogP', 'PSA', 'HBD', 'HBA', 'Rotatable_Bonds', 'Activity (%)']].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
               cbar_kws={"shrink": .6}, ax=ax9)
    plt.title('Property Correlation Matrix', fontsize=14)
    plt.tight_layout()
    images.append((f"9. Property Correlations - Top {n_compounds}", fig_to_buf(fig9)))

    # 9. Scatter plot with regression line for key properties
    fig10, ax10 = plt.subplots(figsize=(8, 6))
    sns.regplot(x='MW', y='LogP', data=df, scatter_kws={'alpha': 0.6, 'color': 'blue'}, 
               line_kws={'color': 'red'}, ax=ax10)
    plt.title('MW vs LogP with Regression Line', fontsize=14)
    plt.xlabel('Molecular Weight', fontsize=12)
    plt.ylabel('LogP', fontsize=12)
    plt.tight_layout()
    images.append((f"10. MW vs LogP Regression - Top {n_compounds}", fig_to_buf(fig10)))

    return images


def build_pdf(df: pd.DataFrame, viz_list: list[tuple]) -> bytes:
    """Assemble PDF with visualisations and molecule pages."""
    pdf = PDF(format="A4", unit="pt")
    pdf.set_auto_page_break(True, margin=40)

    # ---- Title page ----
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.ln(50)
    pdf.multi_cell(0, 30, "Virtual Screening\nAnalysis Report", 0, 'C')
    pdf.ln(20)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 15, f"Number of compounds analyzed: {len(df)}", 0, 1, 'C')
    pdf.cell(0, 15, f"Model Type Used: {MODEL_TYPE_NAME}", 0, 1, 'C')
    pdf.ln(20)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 20, "Summary Statistics", 0, 1, 'L')
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 15, f" - Avg. Predicted Activity: {df['Activity (%)'].mean()*100:.2f}%", 0, 1, 'L')
    pdf.cell(0, 15, f" - Avg. Molecular Weight: {df['MW'].mean():.2f}", 0, 1, 'L')
    pdf.cell(0, 15, f" - Avg. LogP: {df['LogP'].mean():.2f}", 0, 1, 'L')
    pdf.cell(0, 15, f" - Compounds passing Rule-of-5: {df['Drug_Like'].sum()} / {len(df)}", 0, 1, 'L')

    # ---- Visualisation pages (كل 40 جزيء في صفحة) ----
    side_margin = 40                       # يطابق الهامش التلقائي
    for title, img_buf in viz_list:
        pdf.add_page()  # صفحة جديدة لكل صورة

        # بدلاً من pdf.chapter_title(title)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 20, title, ln=1)

        # المساحة المتاحة بعد العنوان
        y_start = pdf.get_y()
        max_height = pdf.h - y_start - side_margin
        max_width  = pdf.w - 2 * side_margin

        # تحميل أبعاد الصورة
        img_buf.seek(0)
        img_w_px, img_h_px = Image.open(img_buf).size
        img_w_pt = img_w_px * 72 / 96
        img_h_pt = img_h_px * 72 / 96

        scale = min(max_width / img_w_pt, max_height / img_h_pt, 1.0)
        new_w = img_w_pt * scale
        new_h = img_h_pt * scale
        x_pos = (pdf.w - new_w) / 2

        img_buf.seek(0)
        pdf.image(img_buf, x=x_pos, y=y_start, w=new_w, h=new_h)

    # ---- Individual molecule pages ----
    for i, row in df.iterrows():
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 25, f"{row['Compound']} (Rank: {row['rank']})", ln=1)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 14, f"SMILES: {row['smiles']}", ln=1)
        pdf.ln(5)

        # رسم الجزيء
        try:
            mol_img = Draw.MolToImage(row["mol"], size=(280, 280))
            m_buf = BytesIO()
            mol_img.save(m_buf, format="PNG")
            m_buf.seek(0)
            pdf.image(m_buf, x=40, y=pdf.get_y(), w=280, h=280)
        except Exception:
            pdf.cell(100, 50, "Image Error", border=1)

        # خصائص الجزيء
        pdf.set_font("Arial", "B", 11)
        prop_x = 340
        prop_y = pdf.get_y()
        pdf.set_xy(prop_x, prop_y)
        pdf.cell(0, 18, "Properties:", ln=1)
        pdf.set_font("Arial", "", 10)
        props = {
            "Activity (%)": f"{row['Activity (%)']*100:.2f}",
            "MW": f"{row['MW']:.2f}",
            "LogP": f"{row['LogP']:.2f}",
            "PSA": f"{row['PSA']:.2f}",
            "HBD / HBA": f"{int(row['HBD'])} / {int(row['HBA'])}",
            "Rot. Bonds": f"{int(row['Rotatable_Bonds'])}",
            "Drug-Like (RO5)": "Yes" if row['Drug_Like'] else "No"
        }
        for k, v in props.items():
            pdf.set_x(prop_x)
            pdf.cell(80, 15, k, border=1)
            pdf.cell(70, 15, v, border=1, ln=1)

    # ---- إخراج الـ PDF كـ bytes ----
    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception as e:
        pdf_buffer = BytesIO()
        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("latin-1")
        pdf_buffer.write(pdf_bytes)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
################################################################################
# Streamlit UI
################################################################################
import streamlit as st
import pandas as pd
import numpy as np
import base64
hide_all_ui = """
    <style>
    /* إخفاء القائمة الجانبية */
    #MainMenu {visibility: hidden;}

    /* إخفاء الفوتر */
    footer {visibility: hidden;}
    footer:after {content:'';}

    /* إخفاء الهيدر القديم */
    header {visibility: hidden;}

    /* إخفاء الهيدر الجديد بكلاس الـ Emotion */
    .st-emotion-cache-h4xjwg {
        display: none !important;
    }
    </style>
"""
custom_hide_css = """
    <style>
    /* إخفاء حاوية البروفايل */
    ._profileContainer_gzau3_53 {
        display: none !important;
    }

    /* إخفاء البادج الخاص بالمشاهد */
    ._container_gzau3_1._viewerBadge_nim44_23 {
        display: none !important;
    }

    /* إخفاء صورة البروفايل بظل */
    ._profileImage_gzau3_78._darkThemeShadow_gzau3_91 {
        display: none !important;
    }

    /* إخفاء الرابط */
    ._link_gzau3_10 {
        display: none !important;
    }

    /* إخفاء العنصر المكرر (لو ظهر مرتين بنفس الكلاسات) */
    ._container_gzau3_1._viewerBadge_nim44_23._container_gzau3_1._viewerBadge_nim44_23 {
        display: none !important;
    }
    </style>
"""
st.markdown(custom_hide_css, unsafe_allow_html=True)
def set_background_with_fade(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            background: linear-gradient(135deg, rgba(16, 20, 31, 0.65), rgba(28, 37, 54, 0.65)),
                        url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            font-family: 'Inter', sans-serif;
        }}
        
        /* Custom CSS for animations and effects */
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        @keyframes gradient {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .animate-fade-in {{
            animation: fadeInUp 0.8s ease-out;
        }}
        
        .pulse-hover:hover {{
            animation: pulse 0.6s ease-in-out;
        }}
        
        .gradient-text {{
            background: linear-gradient(45deg, #00D4AA, #00A8CC, #0084FF);
            background-size: 200% 200%;
            animation: gradient 3s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .glass-card {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }}
        
        .glass-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.4);
            border-color: rgba(0, 212, 170, 0.3);
        }}
        
        .neon-border {{
            border: 2px solid transparent;
            border-radius: 20px;
            background: linear-gradient(45deg, rgba(0, 212, 170, 0.3), rgba(0, 132, 255, 0.3)) border-box;
            -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: subtract;
            mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
            mask-composite: subtract;
        }}
        
        /* Custom button styles */
        .stButton > button {{
            background: linear-gradient(135deg, #00D4AA, #00A8CC);
            border: none;
            border-radius: 16px;
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            padding: 0.8rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(0, 212, 170, 0.4);
            background: linear-gradient(135deg, #00E5BB, #00B8DD);
        }}
        
        /* Enhanced sidebar */
        .css-1d391kg {{
            background: rgba(16, 20, 31, 0.9);
            backdrop-filter: blur(20px);
        }}
        
        /* Custom metric cards */
        .metric-card {{
            background: linear-gradient(135deg, rgba(0, 212, 170, 0.1), rgba(0, 132, 255, 0.1));
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: scale(1.02);
            border-color: rgba(0, 212, 170, 0.3);
        }}
        
        /* Custom tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, #00D4AA, #00A8CC);
            border-color: rgba(0, 212, 170, 0.5);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def show_welcome_page():
    """عرض صفحة الترحيب المحسنة"""
    
    set_background_with_fade("Chemistry Wallpaper2.jpg")
    
    # Hero Section with enhanced design
    st.markdown("""
    <div class="animate-fade-in" style="text-align: center; padding: 3rem 0 2rem 0;">
        <div style="display: inline-block; position: relative;">
            <h1 style="font-size: 5rem; font-weight: 700; margin: 0; letter-spacing: -2px;">
                <span class="gradient-text">Smart</span><span style="color: #DDDDDD;">VEGFR</span>
            </h1>
            <div style="position: absolute; top: -10px; right: -20px; width: 20px; height: 20px; 
                        background: linear-gradient(45deg, #00D4AA, #0084FF); border-radius: 50%; 
                        box-shadow: 0 0 20px rgba(0, 212, 170, 0.6); animation: pulse 2s infinite;"></div>
        </div>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.4rem; font-weight: 300; 
                  margin-top: 1rem; letter-spacing: 1px;">
            Advanced AI-Powered VEGFR Inhibitor Discovery Platform
        </p>
        <div style="width: 100px; height: 3px; background: linear-gradient(90deg, #00D4AA, #0084FF); 
                    margin: 2rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Cards
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Main info card - تم إصلاح المشكلة هنا
        st.markdown("""
        <div class="glass-card animate-fade-in pulse-hover" style="padding: 3rem; margin: 2rem 0;">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div style="display: inline-block; padding: 1rem; background: linear-gradient(135deg, rgba(0, 212, 170, 0.2), rgba(0, 132, 255, 0.2)); 
                           border-radius: 20px; margin-bottom: 1rem;">
                    <span style="font-size: 3rem;">🎯</span>
                </div>
                <h2 style="color: #00D4AA; font-size: 2.2rem; font-weight: 600; margin: 0;">
                    About SmartVEGFR
                </h2>
            </div>
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 1.1rem; line-height: 1.8; text-align: center; margin-bottom: 2rem;">
                SmartVEGFR harnesses cutting-edge machine learning algorithms to predict VEGFR inhibitor activity 
                with unprecedented accuracy. Transform your drug discovery workflow with AI-powered molecular screening.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards grid
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin: 2rem 0;">
            <div class="glass-card" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
                <h3 style="color: #00D4AA; font-size: 1.3rem; margin-bottom: 0.5rem;">AI-Powered</h3>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0;">
                    Advanced ML models trained on extensive VEGFR data
                </p>
            </div>
            <div class="glass-card" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #00A8CC; font-size: 1.3rem; margin-bottom: 0.5rem;">Comprehensive</h3>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0;">
                    Detailed molecular analysis and drug-likeness assessment
                </p>
            </div>
            <div class="glass-card" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📈</div>
                <h3 style="color: #0084FF; font-size: 1.3rem; margin-bottom: 0.5rem;">Visual Reports</h3>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0;">
                    Interactive charts and professional PDF reports
                </p>
            </div>
            <div class="glass-card" style="padding: 2rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
                <h3 style="color: #00D4AA; font-size: 1.3rem; margin-bottom: 0.5rem;">High-Speed</h3>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0;">
                    Rapid screening of large compound libraries
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        
        st.markdown("""
        <div class="glass-card" style="padding: 1.2rem; margin: 1rem 0; text-align: center;">
            <h3 style="color: #00D4AA; font-size: 1.8rem; margin-bottom: 1rem;">
                🚀 How It Works
            </h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #00D4AA, #00A8CC); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        margin-bottom: 0.8rem; font-size: 1.3rem; color: white; font-weight: bold;">1</div>
                <h4 style="color: #00D4AA; margin-bottom: 0.3rem; font-size: 1rem;">Upload</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin: 0; line-height: 1.2;">
                    CSV with SMILES
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #00A8CC, #0084FF); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        margin-bottom: 0.8rem; font-size: 1.3rem; color: white; font-weight: bold;">2</div>
                <h4 style="color: #00A8CC; margin-bottom: 0.3rem; font-size: 1rem;">Configure</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin: 0; line-height: 1.2;">
                    Set parameters
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #0084FF, #6C5CE7); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        margin-bottom: 0.8rem; font-size: 1.3rem; color: white; font-weight: bold;">3</div>
                <h4 style="color: #0084FF; margin-bottom: 0.3rem; font-size: 1rem;">Analyze</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin: 0; line-height: 1.2;">
                    AI prediction
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
            <div style="text-align: center; padding: 0.5rem; display: flex; flex-direction: column; align-items: center;">
                <div style="width: 50px; height: 50px; background: linear-gradient(135deg, #6C5CE7, #A29BFE); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; 
                        margin-bottom: 0.8rem; font-size: 1.3rem; color: white; font-weight: bold;">4</div>
                <h4 style="color: #6C5CE7; margin-bottom: 0.3rem; font-size: 1rem;">Download</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.85rem; margin: 0; line-height: 1.2;">
                    Results & reports
                </p>
            </div>
            """, unsafe_allow_html=True)

        
        # CTA Button
        st.markdown('<div style="text-align: center; margin: 3rem 0;">', unsafe_allow_html=True)
        
        if st.button("🚀 Launch SmartVEGFR Platform", 
                    type="primary", 
                    use_container_width=True,
                    help="Access the main screening interface"):
            st.session_state.show_main_app = True
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer info
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem; margin-top: 2rem; text-align: center;">
            <p style="color: rgba(0, 212, 170, 0.8); margin: 0; font-size: 0.95rem;">
                <strong>⚠️ Research Use Only:</strong> This platform utilizes pre-trained ML models. 
                Results are intended for research and educational purposes.
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_main_app():
    """عرض التطبيق الرئيسي المحسن"""
    
    set_background_with_fade("Chemistry Wallpaper2.jpg")
    
    # Enhanced Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem;">
            <span class="gradient-text">Smart</span><span style="color: #DDDDDD;">VEGFR</span>
        </h1>
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem;">
            AI-Powered VEGFR Inhibitor Screening Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model type name (you'll need to define this variable)
    MODEL_TYPE_NAME = "Random Forest"  # Example - replace with your actual model name
    
    # Enhanced instructions card
    st.markdown(f"""
    <div class="glass-card" style="padding: 2rem; margin-bottom: 2rem;">
        <h3 style="color: #00D4AA; margin-bottom: 1rem; display: flex; align-items: center;">
            <span style="margin-right: 0.5rem;">📋</span> Quick Start Guide
        </h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📁</div>
                <h4 style="color: #00D4AA; font-size: 0.9rem; margin: 0;">1. Upload CSV</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    File with 'smiles' column
                </p>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🔢</div>
                <h4 style="color: #00A8CC; font-size: 0.9rem; margin: 0;">2. Set Top-N</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    Choose compounds (10-50)
                </p>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">🚀</div>
                <h4 style="color: #0084FF; font-size: 0.9rem; margin: 0;">3. Start Analysis</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    Click predict button
                </p>
            </div>
            <div class="metric-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📊</div>
                <h4 style="color: #6C5CE7; font-size: 0.9rem; margin: 0;">4. Download</h4>
                <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0.5rem 0 0 0;">
                    CSV & PDF reports
                </p>
            </div>
        </div>
        <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
            <small style="color: rgba(255, 255, 255, 0.6);">
                Model: {MODEL_TYPE_NAME} | Morgan Fingerprints
            </small>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Welcome Page", help="Return to welcome screen"):
            st.session_state.show_main_app = False
            st.rerun()
    
    st.divider()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="glass-card" style="padding: 0.8rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: center; min-height: 80px;">
            <h3 style="color: #00D4AA; margin: 0; text-align: center; font-size: 1.5rem;">
                ⚙️Configuration
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        file = st.file_uploader("📄 Upload a CSV file containing a 'smiles' column", type="csv", help="Upload a CSV file containing a 'smiles' column")
        
        st.markdown("""
        <style>
        .stButton > button {
            color: white !important;
            font-weight: bold !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
        }

        .stButton > button:hover {
            color: #ffffff !important;
        }

        .stButton > button:disabled {
            color: #CCCCCC !important;
        }
        </style>
        """, unsafe_allow_html=True)

        run = st.button("🚀 Start Prediction & Screening", 
                        disabled=file is None, 
                        type="primary",
                        use_container_width=True)
        
        st.divider()
        
        # Model info card
        st.markdown("""
        <div class="glass-card" style="padding: 1rem;">
            <h4 style="color: #00A8CC; margin-bottom: 0.5rem;">🔬 Model Info</h4>
            <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.8rem; margin: 0;">
                Morgan fingerprints with optimized feature selection for enhanced accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced slider section
    st.markdown("""
    <div class="glass-card" style="padding: 2rem; margin-bottom: 2rem;">
        <h3 style="color: #00D4AA; margin-bottom: 1.5rem; text-align: center;">
            🔢 Select Number of Top Compounds
        </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])

    with col1:
        N_slider = st.slider(
            "Number of top compounds for detailed analysis", 
            min_value=1, 
            max_value=500, 
            value=10,
            help="Select compounds with highest predicted activity scores"
        )

    with col2:
        N_number = st.number_input(
            "Exact Number",
            min_value=1,
            max_value=500,
            value=N_slider,
            step=1
        )

    N = N_number
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Placeholder for main logic - يجب إضافة باقي الكود هنا
    if run and file is not None:
        # Use session state to keep results after run
        st.session_state.run_pressed = True 
        
        try:
            # Load data
            with st.spinner("🔄 Loading and validating data..."):
                df_raw = pd.read_csv(file)
                
                if "smiles" not in df_raw.columns:
                    st.error("❌ Error: CSV file must contain a column named 'smiles'.")
                    st.stop()
                if df_raw.empty:
                    st.error("❌ Error: CSV file is empty.")
                    st.stop()
                
                st.success(f"✅ Data loaded: {len(df_raw)} compounds found")
                
            # Clean and process
            with st.spinner("🧪 Processing molecular structures..."):
                smiles_series = df_raw["smiles"].dropna().drop_duplicates()
                # st.toast(f"Found {len(smiles_series)} unique SMILES")
                X_raw, mols_list, valid_smiles, properties_df = process_smiles(smiles_series.tolist())

                if X_raw is None:
                    st.error("❌ No valid SMILES compounds found in the file.")
                    st.stop()
                
                st.success(f"✅ Processed {len(valid_smiles)} valid molecules")
        
            # Make predictions
            with st.spinner("🤖 Making AI predictions..."):
                # Apply selector and predict
                X_sel = SELECTOR.transform(X_raw)
                if hasattr(MODEL, 'predict_proba'): # Scikit-learn
                   probs = MODEL.predict_proba(X_sel)[:, 1]
                else: # Keras / other
                   probs = MODEL.predict(X_sel).ravel() # .ravel() flattens keras output

                # Create full results DataFrame
                results = pd.DataFrame({
                    "smiles": valid_smiles,
                    "probability": probs,
                    "Activity (%)": probs ,
                    "mol": mols_list,
                     "fp_raw": list(X_raw) # keep raw FP for t-SNE
                     })
                results = pd.concat([results.reset_index(drop=True), properties_df.reset_index(drop=True)], axis=1)
                
                # Sort and select Top N
                top_df = results.sort_values("probability", ascending=False).head(N).reset_index(drop=True)
                top_df['rank'] = top_df['probability'].rank(ascending=False, method='min').astype(int)
                top_df['Compound'] = [f'cpd_{i+1}' for i in range(len(top_df))]
                
                # Store in session state
                st.session_state.top_df = top_df
                st.session_state.fps_array_topN = np.array(top_df['fp_raw'].tolist())
                st.session_state.N = N
            
            st.success(f"🎯 Analysis complete! Found top {len(top_df)} compounds")

        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.stop()

    # --- Display Results if available in session state ---
    if 'top_df' in st.session_state:
        top_df = st.session_state.top_df
        fps_array_topN = st.session_state.fps_array_topN
        N_results = len(top_df) # Actual number retrieved

        # Results summary with enhanced UI
        st.markdown("""
        <div class="glass-card" style="padding: 2rem; margin: 2rem 0;">
            <h2 style="color: #00D4AA; text-align: center; margin-bottom: 1.5rem;">
                🎯 Screening Results Summary
            </h2>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00D4AA; font-size: 2rem; margin-bottom: 0.5rem;">
                    {N_results}
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Top Compounds</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_activity = top_df['Activity (%)'].mean()*100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #00A8CC; font-size: 2rem; margin-bottom: 0.5rem;">
                    {avg_activity:.1f}%
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Avg Activity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            drug_like_count = top_df['Drug_Like'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #0084FF; font-size: 2rem; margin-bottom: 0.5rem;">
                    {drug_like_count}
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Drug-Like</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            max_activity = top_df['Activity (%)'].max()*100
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #6C5CE7; font-size: 2rem; margin-bottom: 0.5rem;">
                    {max_activity:.1f}%
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">Best Hit</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Results tabs with enhanced design
        tab1, tab2, tab3 = st.tabs(["📊 Interactive Results", "📈 Visualizations", "📄 Download Reports"])
        
        with tab1:
            st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; margin: 1rem 0;">
                <h3 style="color: #00D4AA; margin-bottom: 1rem;">
                    🏆 Top Predicted VEGFR Inhibitors
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display top compounds table with enhanced formatting
            display_cols = ['rank','smiles', 'Activity (%)','Drug_Like', 'MW', 'LogP', 'PSA', 'HBD', 'HBA', 'Rotatable_Bonds']
            display_df = top_df[display_cols].round(3).copy()
            display_df.columns = ['Rank', 'SMILES', 'Activity (%)', 'Drug-Like', 'MW', 'LogP', 'PSA', 'HBD', 'HBA', 'Rot. Bonds']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Activity (%)": st.column_config.ProgressColumn(
                        "Activity (%)",
                        help="Predicted VEGFR inhibition activity",
                        min_value=0,
                        max_value=1,
                    ),
                    "SMILES": st.column_config.TextColumn(
                        "SMILES",
                        help="Molecular structure in SMILES format",
                        max_chars=100,
                    ),
                }
            )

        with tab2:
            st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; margin: 1rem 0;">
                <h3 style="color: #00D4AA; margin-bottom: 1rem;">
                    📈 Analysis Visualizations
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Generate Visuals and PDF ---
            with st.spinner("🎨 Generating visualizations..."):
                viz_list = create_all_visualisations(top_df, fps_array_topN)
            st.toast("Visualizations Ready!")

            # --- Preview Visualisations ---
            # Use tabs for better organization
            tab_titles = [item[0] for item in viz_list]
            viz_tabs = st.tabs(tab_titles)
            for i, (title, buf) in enumerate(viz_list):
                with viz_tabs[i]:
                   st.image(buf, caption=title, use_column_width=True)

        with tab3:
            st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; margin: 1rem 0;">
                <h3 style="color: #00D4AA; margin-bottom: 1rem;">
                    📥 Download Analysis Reports
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate PDF
            with st.spinner("📄 Generating PDF report..."):
                if 'viz_list' not in locals():
                    viz_list = create_all_visualisations(top_df, fps_array_topN)
                pdf_bytes = build_pdf(top_df, viz_list)

            # --- Download Buttons ---
            col1, col2 = st.columns(2)
            with col1:
              st.download_button(
                "📊 Download Results (CSV)",
                data=top_df[display_cols].to_csv(index=False).encode('utf-8'),
                file_name=f"top_{N_results}_compounds.csv",
                mime="text/csv",
                use_container_width=True
              )
            with col2:
               st.download_button(
                "📄 Download Comprehensive Report (PDF)",
                data=pdf_bytes,
                file_name=f"screening_report_top_{N_results}.pdf",
                mime="application/pdf",
                 use_container_width=True
               )
            
            # Clear results button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Clear Results & Start New Analysis", type="secondary"):
                for key in ['top_df', 'fps_array_topN', 'N', 'run_pressed']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    elif file is None and not 'top_df' in st.session_state:
        st.markdown("""
        <div class="glass-card" style="padding: 3rem; text-align: center; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📁</div>
            <h3 style="color: #00D4AA; margin-bottom: 1rem;">Ready to Begin Analysis</h3>
            <p style="color: rgba(255, 255, 255, 0.7); font-size: 1.1rem;">
                Upload your CSV file with SMILES data to start the AI-powered screening process
            </p>
        </div>
        """, unsafe_allow_html=True)

# Main App Logic
if 'show_main_app' not in st.session_state:
    st.session_state.show_main_app = False

# عرض الصفحة المناسبة
if st.session_state.show_main_app:
    show_main_app()
else:
    show_welcome_page()
