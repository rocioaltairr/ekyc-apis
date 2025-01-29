from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/face_verification', methods=['POST'])
def post_example():
    input_data = request.get_json()

    data_1 = input_data['image_1']
    data_2 = input_data['image_2']

    result = DeepFace.verify(data_1, data_2, model_name="VGG-Face", detector_backend="opencv", 
                             distance_metric="cosine", enforce_detection=True, align=True, normalization="base")
    
    response_data = {
        "verification_distance": result['distance'],
        "verification_threshold": result['threshold'],
        "verification_result": bool(result['verified'])
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)