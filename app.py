import streamlit as st

st.title("Air Quality Map")

# Tableau Public URL
tableau_public_url = "https://public.tableau.com/views/Book1_17323934199370/Sheet2?:embed=yes&:display_count=no&:showVizHome=no"

# Embed the Tableau Public dashboard
st.components.v1.iframe(src=tableau_public_url, width=1100, height=850)
