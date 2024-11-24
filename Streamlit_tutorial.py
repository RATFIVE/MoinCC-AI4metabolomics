import streamlit as st
import pandas as pd
import plotly.express as px

a = 7

df = pd.DataFrame({
    'Name': [1, 2],
    'Age':  [40, 50]
})


# Beispiel-Daten für die Plotly-Grafik
data = {
    'x': [1, 2, 3, 4, 5],
    'y': [10, 11, 12, 13, 14]
}
df_plot = pd.DataFrame(data)

# Erstellen einer Plotly-Liniendiagramm-Grafik
fig = px.line(df_plot, x='x', y='y', title='Beispiel Plotly Grafik')

def main():
    # SIdebar
    st.sidebar.markdown('Sidebar')
    st.sidebar.dataframe(df)

    # Page
    tab1, tab2 = st.tabs(['Tab1', 'Tab2'])
    with tab1:
        st.title('Tutorial')
        st.markdown(f"""
            # Header
            ## Unterüberschrift
                a =  {a}
            """)
        st.dataframe(df)

    with tab2:
        # Beispiel-Plotly-Grafik erstellen
        st.plotly_chart(fig)

def create_plotly_figure():

    return fig

if __name__ == '__main__':
    main()