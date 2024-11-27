import Layout
import panel3_contour_plot as p3cp
from pathlib import Path
import pandas as pd
from LoadData import *
import streamlit as st



#loaddata = LoadData()

#df_list = loaddata.load_data_list('FA_20231123_2H Yeast_Fumarate-d2_12 .csv')
#df = pd.read_csv(df_list[0])
#print(df)



#example_image_path = str(Path(r'/Users/marco/Documents/MoinCC-AI4metabolomics/app/example/FA_20240205_2H_yeast _Gluc-d2_5.csv_time_dependence.png'))


#fig3 = p3cp.ContourPlot(df=df)
#fig3 = fig3.plot()



def main():
    #st.session_state.clear()
   # st.rerun()
    app = Layout.StreamlitApp()#fig2=example_image_path,
                              #fig3=fig3)
    app.run()


if __name__ == '__main__':
    main()

