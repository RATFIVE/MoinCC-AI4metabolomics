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
    st.session_state.slider1 = 50  # Default value for y1 slider
if "slider2" not in st.session_state:
    st.session_state.slider2 = 500  # Default value for y2 slider
if "last_updated" not in st.session_state:
    st.session_state.last_updated = None  # Tracks which slider was updated

# Callback functions to detect slider changes
def update_slider1():
    st.session_state.last_updated = "slider1"


def update_slider2():
    st.session_state.last_updated = "slider2"


# Layout for sliders and plots
col1, col2 = st.columns(2)

# Slider 1 in col1
with col1:
    st.slider(
        "Adjust Range for y1",
        0,
        100,
        st.session_state.slider1,
        key="slider1",
        on_change=update_slider1,  # Callback to track changes
    )

# Slider 2 inside an expander in col2
with col2:
    with st.expander("Settings"):
        st.slider(
            "Adjust Range for y2",
            0,
            1000,
            st.session_state.slider2,
            key="slider2",
            on_change=update_slider2,  # Callback to track changes
        )

# Create empty placeholders for both plots
plot1_placeholder = col1.empty()
plot2_placeholder = col2.empty()

# Render only the updated plot while preserving the other
if st.session_state.last_updated == "slider1":
    plot1_placeholder.plotly_chart(
        px.line(df, x="x", y="y1", title="Plot 1: y1").update_yaxes(range=[0, st.session_state.slider1]),
        use_container_width=True,
    )
elif st.session_state.last_updated == "slider2":
    plot2_placeholder.plotly_chart(
        px.line(df, x="x", y="y2", title="Plot 2: y2").update_yaxes(range=[0, st.session_state.slider2]),
        use_container_width=True,
    )
else:
    # Render both plots initially
    plot1_placeholder.plotly_chart(
        px.line(df, x="x", y="y1", title="Plot 1: y1").update_yaxes(range=[0, st.session_state.slider1]),
        use_container_width=True,
    )

    plot2_placeholder.plotly_chart(
        px.line(df, x="x", y="y2", title="Plot 2: y2").update_yaxes(range=[0, st.session_state.slider2]),
        use_container_width=True,
    )
