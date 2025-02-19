from flask import Flask, render_template, request
from model import classify_image
from PIL import Image

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return 'No file passed'
    
    file = request.files['image']
    
    if file:
        img = Image.open(io.BytesIO(file.read))
        
        return f"Classification: {classify_image(img)}"
    
    
if __name__ == "__main__":
    app.run(debug=True)
