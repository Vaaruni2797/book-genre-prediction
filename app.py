import os
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

##Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            title=request.form.get('title'),
            summary=request.form.get('summary')
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        results, probabilities = predict_pipeline.predict(pred_df['summary'])
        prob_dict = {}
        for prob, genre in probabilities:
            prob_dict[genre]=prob

        fig = go.Figure([go.Bar(x=list(prob_dict.values()), y=list(prob_dict.keys()),orientation='h')])
        fig.update_layout(title='Predicted Genre Probabilities', 
                        xaxis_title='Genre', 
                        yaxis_title='Probability',
                        yaxis=dict(autorange='reversed'))
        chart_html = pio.to_html(fig, full_html=False)

        return render_template('home.html', results=results[0], chart=chart_html)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))