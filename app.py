import json
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from deepface import DeepFace

class RequestHandler(BaseHTTPRequestHandler):
    
    def _send_response(self, code, content_type, data):
        """發送 HTTP 回應"""
        self.send_response(code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(data.encode())

    def do_POST(self):
        """處理 POST 請求"""
        if self.path == '/face_verification':
            # 讀取 POST 請求的內容
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                # 解析 JSON 數據
                input_data = json.loads(post_data)

                # 取得 Base64 編碼的圖片數據
                data_1 = input_data['image_1']
                data_2 = input_data['image_2']

                # 使用 DeepFace 進行人臉驗證
                result = DeepFace.verify(data_1, data_2, model_name="VGG-Face", detector_backend="opencv", 
                                         distance_metric="cosine", enforce_detection=True, align=True, normalization="base")

                # 準備回應數據
                response_data = {
                    "verification_distance": result['distance'],
                    "verification_threshold": result['threshold'],
                    "verification_result": bool(result['verified'])
                }

                # 發送回應
                self._send_response(200, 'application/json', json.dumps(response_data))

            except Exception as e:
                # 錯誤處理，返回錯誤信息
                error_response = {"error": str(e)}
                self._send_response(400, 'application/json', json.dumps(error_response))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8080):
    """啟動伺服器"""
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
