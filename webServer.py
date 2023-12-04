from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
from PIL import Image
import infer as inferlib

def analyze_image_u2net(img):
    inferlib.infer(
        "model/cloth_segm.pth",
        img,
        cuda=True,
        output_dir="output",
    )


app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nu a fost furnizat niciun fișier'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Numele fișierului este gol'}), 400

    # Aici poți adăuga logica de procesare a fișierului sau pur și simplu returna fișierul
    # În exemplul de mai jos, salvăm fișierul și returnăm numele acestuia
    #file.save('uploaded_image.jpg')
    
    # Aici poți adăuga logica de procesare a fișierului sau pur și simplu returna fișierul
    # În exemplul de mai jos, returnăm conținutul binar al fișierului
    file_content = file.read()

    image = Image.open(file)
    image = image.convert('RGB')
    analyze_image_u2net(image)

    image_path = 'output/seg/final_seg.png'
    return send_file(image_path, mimetype='image/png', as_attachment=True, download_name='imagine.png')

if __name__ == '__main__':
    app.run(debug=True)
