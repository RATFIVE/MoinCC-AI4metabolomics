import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# set the page layout
st.set_page_config(layout="wide")

class StreamlitApp():

    def __init__(self, fig1=None, fig2=None):
        self.fig1 = fig1  # Removed the trailing comma
        self.fig2 = fig2  # Removed the trailing comma

    def side_bar(self):
        st.sidebar.title('Side Panel')
        st.sidebar.markdown('This is the side panel')

        # Beispiel-Widgets im Seitenpanel
        slider_value = st.sidebar.slider("Select a value", 0, 100, 50)
        st.sidebar.write(f"Slider value: {slider_value}")


    def header(self):
        st.markdown(
            """
                # MoinCC - Application
            """
        )

    def columns(self):
        col1, col2 = st.columns([2, 3])

        
        with col1:
            st.markdown("""
        ## Sine Wave Plot
        This is a simple Plotly figure showing a sine wave. The plot displays the relationship 
        between the x-axis and y-axis, where the y-axis represents the sine of the values on the x-axis.
        The sine wave oscillates between -1 and 1.
                        """)
            

        with col2:
            st.plotly_chart(self.fig1)



# d

    

    def run(self):
        self.side_bar()
        self.header()
        self.columns()



# Create a simple Plotly figure
def create_plot():
    x = np.linspace(0, 10, 1000)
    y = np.log(x)

    fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Sine Wave'))
    fig.update_layout(title="Sine Wave Example", xaxis_title="X", yaxis_title="Y")
    return fig

if __name__ == '__main__':

    fig1 = create_plot()

    # Run Streamlit App
    app = StreamlitApp(fig1)
    app.run()
