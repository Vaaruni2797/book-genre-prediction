import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import plotly.graph_objs as go

# Streamlit App
st.title("Book Genre Prediction")

# Initialize session state for storing probabilities
if 'prob_dict' not in st.session_state:
    st.session_state.prob_dict = None

# Create form for user input
with st.form(key='predict_form'):
    title = st.text_input("Enter your book title")
    summary = st.text_area("Enter your book summary")
    submit_button = st.form_submit_button(label='Predict your Book Genre')

# When form is submitted, run prediction
if submit_button:
    if title and summary:
        # Prepare data for prediction
        data = CustomData(title=title, summary=summary)
        pred_df = data.get_data_as_data_frame()

        # Predict genre
        predict_pipeline = PredictPipeline()
        results, probabilities = predict_pipeline.predict(pred_df['summary'])

        # Create a dictionary of genre and probabilities
        prob_dict = {genre: prob for prob, genre in probabilities}

        # Store the probabilities in session state
        st.session_state.prob_dict = prob_dict

        # Display the predicted genre
        st.write(f"### Predicted Genre: **{results[0].upper()}**")

# Button to display detailed analysis (only shown after prediction)
if st.session_state.prob_dict:
    if st.button('See Detailed Analysis'):
        prob_dict = st.session_state.prob_dict

        # Create tabs for Bar Chart and Radar Chart
        tab1, tab2 = st.tabs(["🌀 Radar Chart","📊 Bar Chart"])

        # Tab 1: Bar Chart
        with tab1:
            
            # Prepare radar chart data
            radar_fig = go.Figure()

            radar_fig.add_trace(go.Scatterpolar(
                r=list(prob_dict.values()),
                theta=list(prob_dict.keys()),
                fill='toself'
            ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title='Radar Chart of Genre Probabilities'
            )

            st.plotly_chart(radar_fig)

        # Tab 2: Radar Chart
        with tab2:
            # Plot bar chart with Plotly
            fig = go.Figure([go.Bar(
                x=list(prob_dict.values()),
                y=list(prob_dict.keys()),
                orientation='h'
            )])

            fig.update_layout(
                title='Predicted Genre Probabilities',
                xaxis_title='Probability',
                yaxis_title='Genre',
                yaxis=dict(categoryorder='total ascending')
            )
            st.plotly_chart(fig)
