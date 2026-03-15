import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="AutoAnalyst AI", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>
.stApp {background-color:#0E1117;}
h1,h2,h3 {color:#00C4B4;}
[data-testid="stMetricValue"] {color:white;}
section[data-testid="stSidebar"] {background:#111827;}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div style="background:#111827;padding:25px;border-radius:10px">
<h1 style="color:#00C4B4">🚀 AutoAnalyst AI</h1>
<p style="color:white">AI Powered Data Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

file = st.file_uploader("Upload Dataset", type=["csv","xlsx"])

if file:

    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    numeric_cols = df.select_dtypes(include=["int64","float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # ---------- SIDEBAR ----------
    st.sidebar.header("Dashboard Controls")

    metric = st.sidebar.selectbox("Metric", numeric_cols)
    category = st.sidebar.selectbox("Category", cat_cols)

    # ---------- TABS ----------
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "Overview",
        "AI Analysis",
        "Ask Analyst",
        "Prediction",
        "Strategy Report"
    ])

    # ---------- DASHBOARD ----------
    with tab1:

        st.subheader("Key Metrics")

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Rows",df.shape[0])
        col2.metric("Columns",df.shape[1])
        col3.metric("Numeric",len(numeric_cols))
        col4.metric("Categorical",len(cat_cols))

        fig1 = px.histogram(
            df,
            x=metric,
            template="plotly_dark",
            color_discrete_sequence=["#00C4B4"]
        )

        grouped = df.groupby(category)[metric].sum().reset_index()

        fig2 = px.bar(
            grouped,
            x=category,
            y=metric,
            template="plotly_dark",
            color_discrete_sequence=["#00C4B4"]
        )

        col1,col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1,use_container_width=True)

            mean_val=df[metric].mean()
            max_val=df[metric].max()
            min_val=df[metric].min()

            st.markdown(f"""
📊 **Insight**

This histogram shows distribution of **{metric}**.

• Average value: **{round(mean_val,2)}**  
• Minimum value: **{round(min_val,2)}**  
• Maximum value: **{round(max_val,2)}**
""")

        with col2:
            st.plotly_chart(fig2,use_container_width=True)

            top_category = grouped.sort_values(metric,ascending=False).iloc[0][category]

            st.markdown(f"""
📊 **Insight**

This bar chart compares **{metric}** across **{category}**.

• Highest contributor: **{top_category}**
""")

        if len(numeric_cols)>1:

            corr=df[numeric_cols].corr()

            fig3=px.imshow(corr,text_auto=True)

            st.plotly_chart(fig3,use_container_width=True)

            st.markdown("""
📊 **Insight**

The correlation heatmap shows relationships between numeric variables.

Values close to **1** indicate strong positive relationships while values near **0** indicate weak relationships.
""")

    # ---------- AI INSIGHTS ----------
    with tab2:

        st.subheader("AI Insights")

        if st.button("Generate Insights"):

            summary=df.describe().to_string()

            prompt=f"""
            Analyze dataset and explain:

            key trends
            opportunities
            risks
            strategies

            {summary}
            """

            response=client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(response.choices[0].message.content)

    # ---------- ASK AI ----------
    with tab3:

        question=st.text_input(
            "Ask something about dataset",
            placeholder="Example: Which region has highest sales?"
        )

        if question:

            summary=df.describe().to_string()

            prompt=f"""
            Dataset summary:
            {summary}

            Question:
            {question}
            """

            response=client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(response.choices[0].message.content)

    # ---------- PREDICTION ----------
    with tab4:

        st.subheader("Regression Prediction")

        feature=st.selectbox("Feature",numeric_cols)
        target=st.selectbox("Target",numeric_cols)

        X=df[[feature]].values
        y=df[target].values

        model=LinearRegression()
        model.fit(X,y)

        future=np.array([[X.max()+1]])

        prediction=model.predict(future)

        st.write("Predicted value:",round(prediction[0],2))

        st.markdown(f"""
📊 **Prediction Insight**

Based on linear regression, the next estimated value of **{target}** is **{round(prediction[0],2)}**.
""")

        st.subheader("Anomaly Detection")

        model=IsolationForest()

        df["anomaly"]=model.fit_predict(df[numeric_cols])

        anomalies=df[df["anomaly"]==-1]

        st.write("Anomalies detected:",len(anomalies))

        st.dataframe(anomalies.head())

        st.markdown(f"""
📊 **Anomaly Insight**

The model detected **{len(anomalies)} unusual records** that differ from normal data patterns.
""")

    # ---------- REPORT ----------
    with tab5:

        st.subheader("Complete Analysis Report")

        st.write("### Dataset Overview")

        st.write(df.describe())

        if st.button("Generate AI Report"):

            summary=df.describe().to_string()

            prompt=f"""
            Write professional business report including:

            dataset overview
            key findings
            strategic recommendations

            {summary}
            """

            response=client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(response.choices[0].message.content)

else:

    st.info("Upload dataset to begin analysis")
