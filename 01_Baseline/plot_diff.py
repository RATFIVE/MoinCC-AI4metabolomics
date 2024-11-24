import streamlit as st
import plotly.express as px
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "x": range(10),
    "y1": [i**2 for i in range(10)],
    "y2": [i**3 for i in range(10)],
})

# Initialize session state for sliders
if "slider1" not in st.session_state:
    st.session_state.slider1 = 0

if "slider2" not in st.session_state:
    st.session_state.slider2 = 0

# Layout for sliders and plots
col1, col2 = st.columns(2)

with col1:
    slider1 = st.slider("Adjust Range for y1", 0, 100, st.session_state.slider1, key="slider1")

with col2:
    slider2 = st.slider("Adjust Range for y2", 0, 1000, st.session_state.slider2, key="slider2")

# Track which slider changed
if slider1 != st.session_state.slider1:
    st.session_state.slider1 = slider1
    st.write("Updating Plot 1...")
    fig1 = px.line(df, x="x", y="y1", title="Plot 1: y1")
    fig1.update_yaxes(range=[0, slider1])
    st.plotly_chart(fig1)

elif slider2 != st.session_state.slider2:
    st.session_state.slider2 = slider2
    st.write("Updating Plot 2...")
    fig2 = px.line(df, x="x", y="y2", title="Plot 2: y2")
    fig2.update_yaxes(range=[0, slider2])
    st.plotly_chart(fig2)
