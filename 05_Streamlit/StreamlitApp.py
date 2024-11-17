import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class StreamlitApp():

    def __init__(self, fig1, fig2):
        self.fig1 = fig1  # Removed the trailing comma
        self.fig2 = fig2  # Removed the trailing comma

    def side_bar(self):
        st.sidebar.title('Side Panel')
        st.sidebar.markdown('This is the side panel')

        # Beispiel-Widgets im Seitenpanel
        slider_value = st.sidebar.slider("Select a value", 0, 100, 50)
        st.sidebar.write(f"Slider value: {slider_value}")
        
        checkbox_value = st.sidebar.checkbox("Check me")
        st.sidebar.write(f"Checkbox is {'checked' if checkbox_value else 'unchecked'}")
        
        selectbox_value = st.sidebar.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])
        st.sidebar.write(f"Selected option: {selectbox_value}")

    

    def run(self):
        self.side_bar()



if __name__ == '__main__':
    # Example DataFrame
    df = pd.DataFrame({
        "x": np.random.randn(100),
        "y": np.random.randn(100)
    })
    
    # First Plot: Scatter Plot
    fig1 = go.Figure(data=[
        go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name='Example')
    ])
    fig1.update_layout(title="Scatter Plot", xaxis_title="X-Axis", yaxis_title="Y-Axis")

    # Second Plot: Scatter Plot (Duplicate for example)
    fig2 = go.Figure(data=[
        go.Scatter(x=df['x'], y=df['y'], mode='lines+markers', name='Example')
    ])
    fig2.update_layout(title="Scatter Plot 2", xaxis_title="X-Axis", yaxis_title="Y-Axis")

    # Run Streamlit App
    app = StreamlitApp(fig1, fig2)
    app.run()
