import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
from PIL import Image
import base64
import io

model = load_model('test_model.h5')

class EnhancedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(EnhancedEncoder, self).default(obj)

class Server(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        raw_body = self.rfile.read(content_length)
        post_data = json.loads(raw_body)
        img_base64 = post_data['image']

        img_byte = base64.b64decode(img_base64)
        img_pil = Image.open(io.BytesIO(img_byte))
        img_pil.thumbnail((150,150), Image.ANTIALIAS)

        img_tensor = image.img_to_array(img_pil)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        preds = model.predict(img_tensor).ravel()
        
        result = dict(
        	chinright=preds.take(0), 
        	onlaptop=preds.take(1), 
        	normal=preds.take(2)
        )
        result_json = json.dumps(result, cls=EnhancedEncoder)

        self._set_headers()
        self.wfile.write(bytes(result_json, 'utf-8'))

def run(server_class=HTTPServer, handler_class=Server, port=8008):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    print('Starting httpd on port %d...' % port)
    httpd.serve_forever()


if __name__ == "__main__":
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()