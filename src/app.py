# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency, ttest_ind
import numpy as np

# Page config
st.set_page_config(page_title="⚛️ Project week 19: Periodic Table and Material Project Database (API)", layout="wide", page_icon="⚛️")

# Custom CSS 
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-weight: 900;
        font-size: 3.5rem;
        margin: 0;
    }
    .main-header p {
        font-size: 2.2rem;
        margin-top: 0.5rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
        border-left: 8px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    .card:hover {
        transform: translateY(-3px);
    }
    .card h3 {
        color: #2d3748;
        font-weight: 700;
        margin-top: 0;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .hypothesis-card {
        background-color: #fff7e6;
        border-left-color: #f59e0b;
    }
    .result-card {
        background-color: #e6f7e6;
        border-left-color: #48bb78;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stExpander {
        border: none;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/PROCESSED_DATA/cleaned_periodic_table.csv')
    # Convert numeric columns
    numeric_cols = ['Atomic Number', 'Atomic_Mass', 'Melting_Point', 'Boiling_Point',
                    'Density', 'Atomic_Radius', 'Electronegativity', 'Ionization_Energy', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

# Helper: get numeric electronegativity (handles duplicate columns)
def get_numeric_electronegativity(df):
    col = df['Electronegativity']
    if isinstance(col, pd.DataFrame):
        return col.select_dtypes(include=[np.number]).iloc[:, 0]
    return col

# Metal/Non-metal categories
metal_cats = ['Alkali metal', 'Alkaline earth', 'Transition metal', 
              'Post-transition metal', 'Lanthanide', 'Actinide']
nonmetal_cats = ['Nonmetal', 'Noble gas', 'Halogen', 'Metalloid', 
                 'Halogen (predicted)', 'Noble gas (predicted)']
df['metal_binary'] = df['Metallic_Character'].apply(
    lambda x: 'Metal' if x in metal_cats else ('Non-metal' if x in nonmetal_cats else 'Unknown')
)

# Header
st.markdown("""
<div class="main-header">
    <h1>⚛️ Project: Periodic Table and Material Project Database (API)</h1>
    <p>Data from periodic table + Materials Project API | Hypothesis Testing & Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("", [
    "📊 Data Overview",
    "📈 Exploratory Data Analysis",
    "🔬 Hypothesis Tests",
    "📝 Conclusions"
])

# Custom CSS for sidebar background and font sizes
st.markdown("""
    <style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #D4E6F1;
    }
    /* Sidebar title (h1) */
    section[data-testid="stSidebar"] h1 {
        font-size: 3.2rem !important;
        font-weight: bold;
    }
    /* Sidebar radio button labels */
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 3.9rem !important;
        font-weight: bold;
        padding: 0.5rem 0;
    }
    /* Optional: Adjust spacing between radio options */
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ----- Data Overview -----
if section == "📊 Data Overview":
    st.header("📊 Data Overview")
    st.markdown("""
    <div class="card">
    <h3>📁 Sources</h3>
    <ul>
        <li><b>Periodic table dataset</b> (35 properties: atomic number, melting point, density, electronegativity, ionization energy, etc.)</li>
        <li><b>Materials Project API</b> (crystal symmetry, unit cell volume, list of compounds containing the element)</li>
    </ul>
    <h3>🧹 Data Cleaning</h3>
    <ul>
        <li>Converted numeric columns, handled missing values by linear interpolation along atomic number order</li>
        <li>Standardized categorical text, removed duplicate columns</li>
        <li>Final cleaned dataset: <code>118 rows × 36 columns</code></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Elements", df.shape[0])
    col2.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📋 Sample of Cleaned Data")
    st.dataframe(df.head(10))

    st.subheader("❓ Missing Values per Column")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ['Column', 'Missing Count']
    st.dataframe(missing_df)

# ----- EDA -----
elif section == "📈 Exploratory Data Analysis":
    st.header("📈 Exploratory Data Analysis")

    st.subheader("🔗 Correlation Matrix")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Distribution Plots")
    plot_cols = st.multiselect("Select numeric columns to visualize", numeric_df.columns.tolist(),
                               default=['Atomic Number', 'Atomic_Mass', 'Melting_Point', 'Density', 'volume'])
    if plot_cols:
        n_cols = min(3, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows*n_cols > 1 else [axes]
        for i, col in enumerate(plot_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("📦 Boxplots (Outlier Visualization)")
    outlier_cols = st.multiselect("Select columns for boxplots", numeric_df.columns.tolist(),
                                  default=['Melting_Point', 'Density', 'volume', 'Atomic_Radius'])
    if outlier_cols:
        fig, axes = plt.subplots(1, len(outlier_cols), figsize=(5*len(outlier_cols), 5))
        if len(outlier_cols) == 1:
            axes = [axes]
        for i, col in enumerate(outlier_cols):
            sns.boxplot(y=df[col].dropna(), ax=axes[i])
            axes[i].set_title(col, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

# ----- Hypothesis Tests -----
elif section == "🔬 Hypothesis Tests":
    st.header("🔬 Hypothesis Testing")

    # Hypothesis 1
    with st.expander("**Hypothesis 1:** Correlation between Atomic Number and Volume", expanded=True):
        data1 = df[['Atomic Number', 'volume']].dropna()
        corr, p = pearsonr(data1['Atomic Number'], data1['volume'])
        st.markdown(f"""
        <div class="card">
        <h3>📐 Test Result</h3>
        <b>Pearson correlation:</b> {corr:.3f}<br>
        <b>p-value:</b> {p:.3e}<br>
        <b>Conclusion:</b> {"✅ Reject H₀ – significant correlation" if p < 0.05 else "❌ Fail to reject H₀ – no significant correlation"}
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.scatter(data1['Atomic Number'], data1['volume'], alpha=0.6, c='#667eea')
        ax.set_xlabel("Atomic Number")
        ax.set_ylabel("Volume (Å³)")
        ax.set_title("Atomic Number vs Volume", fontweight='bold')
        st.pyplot(fig)

    # Hypothesis 2
    with st.expander("**Hypothesis 2:** Metallic Character vs. Magnetism Type"):
        df2 = df[df['metal_binary'] != 'Unknown']
        cont = pd.crosstab(df2['metal_binary'], df2['Magnetism Type'])
        if cont.shape[0] > 1 and cont.shape[1] > 1:
            chi2, p, dof, ex = chi2_contingency(cont)
            st.markdown(f"""
            <div class="card">
            <h3>📐 Test Result</h3>
            <b>Chi‑square:</b> {chi2:.3f}<br>
            <b>p-value:</b> {p:.3e}<br>
            <b>Conclusion:</b> {"✅ Reject H₀ – association exists" if p < 0.05 else "❌ Fail to reject H₀ – no association"}
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(cont)
        else:
            st.warning("Insufficient categories for Chi‑square test.")

    # Hypothesis 3
    with st.expander("**Hypothesis 3:** Volume Difference (Metals vs. Non‑metals)"):
        metals = df[df['metal_binary'] == 'Metal']['volume'].dropna()
        nonmetals = df[df['metal_binary'] == 'Non-metal']['volume'].dropna()
        if len(metals) > 1 and len(nonmetals) > 1:
            t_stat, p_val = ttest_ind(metals, nonmetals, equal_var=False)
            st.markdown(f"""
            <div class="card">
            <h3>📐 Test Result</h3>
            <b>t‑statistic:</b> {t_stat:.3f}<br>
            <b>p-value:</b> {p_val:.3e}<br>
            <b>Conclusion:</b> {"✅ Reject H₀ – significant difference" if p_val < 0.05 else "❌ Fail to reject H₀ – no significant difference"}
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[df['metal_binary'].isin(['Metal','Non-metal'])]['metal_binary'], 
                        y=df[df['metal_binary'].isin(['Metal','Non-metal'])]['volume'], ax=ax, palette='Set2')
            ax.set_xlabel("Metal / Non‑metal")
            ax.set_ylabel("Volume (Å³)")
            ax.set_title("Volume Comparison", fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for t‑test.")

    # Hypothesis 4 (Ionization Energy vs Electronegativity correlation)
    with st.expander("**Hypothesis 4:** Positive Correlation between Ionization Energy and Electronegativity"):
        # Use numeric conversion and dropna
        corr_data = df[['Ionization_Energy', 'Electronegativity']].apply(pd.to_numeric, errors='coerce').dropna()
        if len(corr_data) > 1:
            corr, p = pearsonr(corr_data['Ionization_Energy'], corr_data['Electronegativity'])
            st.markdown(f"""
            <div class="card">
            <h3>📐 Test Result</h3>
            <b>Pearson correlation:</b> {corr:.3f}<br>
            <b>p-value:</b> {p:.3e}<br>
            <b>Conclusion:</b> {"✅ Reject H₀ – significant positive correlation" if p < 0.05 and corr > 0 else "❌ Fail to reject H₀ – no significant positive correlation"}
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.scatter(corr_data['Ionization_Energy'], corr_data['Electronegativity'], alpha=0.6, c='#f59e0b')
            ax.set_xlabel("Ionization Energy (eV)")
            ax.set_ylabel("Electronegativity")
            ax.set_title("Ionization Energy vs Electronegativity", fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for correlation test.")

    # Hypothesis 5 (Ionization Energy difference)
    with st.expander("**Hypothesis 5:** Ionization Energy Difference (Metals vs. Non‑metals)"):
        metals_ion = df[df['metal_binary'] == 'Metal']['Ionization_Energy'].dropna()
        nonmetals_ion = df[df['metal_binary'] == 'Non-metal']['Ionization_Energy'].dropna()
        if len(metals_ion) > 1 and len(nonmetals_ion) > 1:
            t_stat, p_val = ttest_ind(metals_ion, nonmetals_ion, equal_var=False)
            st.markdown(f"""
            <div class="card">
            <h3>📐 Test Result</h3>
            <b>t‑statistic:</b> {t_stat:.3f}<br>
            <b>p-value:</b> {p_val:.3e}<br>
            <b>Conclusion:</b> {"✅ Reject H₀ – significant difference" if p_val < 0.05 else "❌ Fail to reject H₀ – no significant difference"}
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[df['metal_binary'].isin(['Metal','Non-metal'])]['metal_binary'], 
                        y=df[df['metal_binary'].isin(['Metal','Non-metal'])]['Ionization_Energy'], ax=ax, palette='Set3')
            ax.set_xlabel("Metal / Non‑metal")
            ax.set_ylabel("Ionization Energy (eV)")
            ax.set_title("Ionization Energy Comparison", fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for t‑test.")

    # Hypothesis 6 (Electronegativity difference)
    with st.expander("**Hypothesis 6:** Electronegativity Difference (Metals vs. Non‑metals)"):
        elec = get_numeric_electronegativity(df)
        df_elec = pd.DataFrame({
            'metal_binary': df['metal_binary'],
            'Electronegativity': elec
        }).dropna()
        metals_elec = df_elec[df_elec['metal_binary'] == 'Metal']['Electronegativity']
        nonmetals_elec = df_elec[df_elec['metal_binary'] == 'Non-metal']['Electronegativity']
        if len(metals_elec) > 1 and len(nonmetals_elec) > 1:
            t_stat, p_val = ttest_ind(metals_elec, nonmetals_elec, equal_var=False)
            st.markdown(f"""
            <div class="card">
            <h3>📐 Test Result</h3>
            <b>t‑statistic:</b> {t_stat:.3f}<br>
            <b>p-value:</b> {p_val:.3e}<br>
            <b>Conclusion:</b> {"✅ Reject H₀ – significant difference" if p_val < 0.05 else "❌ Fail to reject H₀ – no significant difference"}
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x=df_elec['metal_binary'], y=df_elec['Electronegativity'], ax=ax, palette='Set1')
            ax.set_xlabel("Metal / Non‑metal")
            ax.set_ylabel("Electronegativity")
            ax.set_title("Electronegativity Comparison", fontweight='bold')
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for t‑test.")

    # Regression (atomic number vs volume)
    with st.expander("📈 Regression: Predicting Volume from Atomic Number"):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        X = df[['Atomic Number']].dropna()
        y = df.loc[X.index, 'volume']
        if len(X) > 1:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            st.markdown(f"""
            <div class="card">
            <h3>📐 Model Summary</h3>
            <b>R²:</b> {r2:.3f}<br>
            <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
            <b>Intercept:</b> {model.intercept_:.3f}
            </div>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            ax.scatter(X, y, alpha=0.5, c='#48bb78')
            ax.plot(X, y_pred, color='red')
            ax.set_xlabel("Atomic Number")
            ax.set_ylabel("Volume (Å³)")
            ax.set_title("Linear Regression", fontweight='bold')
            st.pyplot(fig)

# ----- Conclusions -----
else:
    st.header("📝 Conclusions")
    st.markdown("""
    <div class="card">
    <h3>✅ Key Findings</h3>
    <ul>
        <li><b>Hypothesis 1:</b> No significant correlation between atomic number and unit cell volume (p > 0.05). Volume depends more on crystal structure.</li>
        <li><b>Hypothesis 2:</b> Metallic character and magnetism type are significantly associated (p < 0.05).</li>
        <li><b>Hypothesis 3:</b> No significant difference in volume between metals and non‑metals (p = 0.07).</li>
        <li><b>Hypothesis 4:</b> Strong positive correlation between ionization energy and electronegativity (p < 5.843e-17).</li>
        <li><b>Hypothesis 5 & 6:</b> Significant differences in ionization energy and electronegativity between metals and non‑metals (both p < 0.001).</li>
        <li><b>Regression:</b> Volume is poorly predicted by atomic number (R² ≈ 0.01).</li>
    </ul>
    </div>

    <div class="card">
    <h3>💡 Further Work</h3>
    <ul>
        <li>Adding the XPS data in the dataset</li>
        <li>Adding the XAS data in the dataset</li>
    </ul>
    </div>

    <div class="card">
    <h3>🚀 Recommendations</h3>
    <ul>
        <li>We need an enriched dataset for educational tools or materials science research.</li>
    </ul>
    </div>

    <div class="card">
    <h3>🔗 Links</h3>
    <ul>
        <li><a href="https://github.com/aaghafari-dev/project_week19.git">My GitHub Repository</a></li>
        <li><a href="https://next-gen.materialsproject.org/">The Materials Project </a></li>
        <li><a href="https://https://www.kaggle.com/datasets/saurabhkumar0101011/all-elements-dataset-hog-periodic-table">Periodic Table from Kaggle</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)