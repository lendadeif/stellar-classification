import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from feature_engineering import apply_feature_engineering

pipeline = joblib.load('final_gb_model1.pkl')
app = Flask(__name__)
class_images = {
    "STAR": r"static/images/star.jpg",
    "GALAXY": r"static/images/galaxy.jpg",
    "QUASAR": r"static/images/qso.jpg",
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_features = {
                "alpha": float(request.form['alpha']),
                "delta": float(request.form['delta']),
                "u": float(request.form['u']),
                "g": float(request.form['g']),
                "r": float(request.form['r']),
                "i": float(request.form['i']),
                "z": float(request.form['z']),
                "redshift": float(request.form['redshift']),
                "plate": float(request.form['plate']),
                # "MJD": float(request.form['mjd']),
                # "fiber_ID": float(request.form['fiberID'])
            }
            input_df = pd.DataFrame([input_features])
            input_df = apply_feature_engineering(input_df)

            # Transforming
            qt = joblib.load('quantile_transformer.pkl')
            features_to_transform = ['alpha', 'delta',
                                     'redshift', 'u_g', 'g_r', 'r_i', 'i_z']
            input_df[features_to_transform] = qt.transform(
                input_df[features_to_transform])
            prediction = pipeline.predict(input_df)[0]
            if prediction == 0:
                prediction = "GALAXY"
            elif prediction == 1:
                prediction = "QSO"
            elif prediction == 2:
                prediction = "STAR"
            if prediction.upper() == "QSO":
                prediction = "QUASAR"
            print(prediction)
            image_file = class_images.get(
                prediction.upper(), "/static/images/default.jpg")
            print(image_file)

            return jsonify({
                'prediction_text': f'The predicted class is: {prediction}',

                'image_file': image_file,
            })
        except Exception as e:
            return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
