from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

@app.route('/face_verification', methods=['POST'])
def post_example():
    # 从请求中获取 JSON 数据
    input_data = request.get_json()
    param1 = input_data['webcam_img']  # 前端传递的 webcam 图像（base64 编码）
    param2 = input_data['name_in_ID']  # 假设这是你从前端传递的 ID 名称或图像标识

    # 在这个例子中，我们不再使用 MongoDB，而是直接从 param2 提供另一个 base64 编码的图像
    # 假设 param2 代表另一张已经在前端或其他地方预处理并传送的 base64 编码图像

    # 注意：param1 和 param2 都应为 base64 编码的图像字符串
    data_1 = param1  # webcam 图像的 base64 编码
    data_2 = param2  # 目标 ID 图像的 base64 编码

    # 使用 DeepFace 进行人脸验证
    result = DeepFace.verify(data_1, data_2, model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine", enforce_detection=True, align=True, normalization="base")
    
    # 返回结果作为 JSON 响应
    response_data = {
        "verification_distance": result['distance'],
        "verification_threshold": result['threshold'],
        "verification_result": bool(result['verified'])
    }
    
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
