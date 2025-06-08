

from flask import Flask, render_template, request
import os
from style_transfer import run_style_transfer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['RESULT_FOLDER'] = 'static/results/'

# Auto-create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transfer', methods=['POST'])
def transfer():
    content_file = request.files['content']
    style_file = request.files['style']

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_file.filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_file.filename)

    # Save uploaded files
    content_file.save(content_path)
    style_file.save(style_path)

    # Define output path
    output_path = os.path.join(app.config['RESULT_FOLDER'], 'output.png')

    # Run style transfer
    run_style_transfer(content_path, style_path, output_path)

    # Show result
    return render_template('result.html', result_image=output_path)

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, render_template, request, redirect, url_for
# import os
# from style_transfer import run_style_transfer

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# app.config['RESULT_FOLDER'] = 'static/results/'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/transfer', methods=['POST'])
# def transfer():
#     content_file = request.files['content']
#     style_file = request.files['style']
#     content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_file.filename)
#     style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_file.filename)
#     content_file.save(content_path)
#     style_file.save(style_path)

#     output_path = os.path.join(app.config['RESULT_FOLDER'], 'output.png')
#     run_style_transfer(content_path, style_path, output_path)

#     return render_template('result.html', result_image=output_path)

# if __name__ == '__main__':
#     app.run(debug=True)
