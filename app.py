import datetime
from flask import Flask, request, render_template, send_from_directory, send_file, jsonify
import os
import pandas as pd
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from insight_generation import identify_patterns
from data_preproceesing import preprocess_data

# Assuming these functions are defined in the corresponding modules
# from data_preprocessing import preprocess_data
# from pattern_identification import identify_patterns

app = Flask(__name__, static_url_path='/static')

# Ensure the directories for static files, preprocessed data, and plots exist
os.makedirs('static', exist_ok=True)
os.makedirs('preprocessed', exist_ok=True)
os.makedirs('plots', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static', filename)
            file.save(file_path)

            # Get selected parameters from the form
            x_param = request.form['firstParameter']
            y_param = request.form['secondParameter']

            # Generate insights graph
            graph_path = generate_insights_graph(file_path, x_param, y_param)
            return jsonify({'graph_url': graph_path})

    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file)

        # Preprocess data
        df_normalized = preprocess_data(df)

        # Save preprocessed data
        preprocessed_file_path = os.path.join('preprocessed', 'preprocessed_data.csv')
        df_normalized.to_csv(preprocessed_file_path, index=False)

        # Identify patterns
        insights_file_path = 'insights.txt'
        identify_patterns(df_normalized, insights_file_path)

        # Load insights
        with open(insights_file_path, 'r') as f:
            insights = f.read()

        return jsonify({'insights': insights})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/plot/<string:plot_name>')
def serve_plot(plot_name):
    plot_file_path = os.path.join('plots', f'{plot_name}_plot.png')
    if os.path.exists(plot_file_path):
        return send_file(plot_file_path, mimetype='image/png')
    else:
        return jsonify({'error': 'Plot not found'}), 404

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

def generate_insights_graph(filename, x_param, y_param):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(filename, encoding='latin1')
    # Group the data by x_param and calculate the average of y_param for each group
    average_data = data.groupby(x_param)[y_param].mean().reset_index()
    # Plot the graph
    plt.plot(average_data[x_param], average_data[y_param], marker='o', linestyle='-')
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f'Average {y_param} by {x_param}')
    plt.grid(True)
    # Save the graph directly in the static directory
    graph_filename = generate_unique_filename('insights_graph.png')
    graph_path = os.path.join('static', graph_filename)
    plt.savefig(graph_path)
    plt.close()
    return graph_path

def generate_unique_filename(filename):
    return f"{filename.split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"

if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug=True)
