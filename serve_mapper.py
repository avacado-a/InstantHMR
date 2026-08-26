"""
Lightweight Web Server for 3D COCO Keypoint Mapping Tool
=========================================================
Serves the Three.js 3D mapper on http://localhost:8088
and automatically saves any custom point mappings back into Python.
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8088
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

class MappingHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve from REPO_ROOT
        clean_path = path.split("?")[0].split("#")[0]
        if clean_path == "/" or clean_path == "/index.html":
            return os.path.join(REPO_ROOT, "coco_mapper.html")
        elif clean_path == "/sample_joints.json":
            return os.path.join(REPO_ROOT, "sample_joints.json")
        return super().translate_path(path)

    def do_POST(self):
        if self.path == "/save_mapping":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode("utf-8"))

            out_path = os.path.join(REPO_ROOT, "instanthmr", "coco_mapping.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\n✓ Successfully saved custom COCO mapping to {out_path}!")
            print(json.dumps(data, indent=2))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "saved_to": "instanthmr/coco_mapping.json"}')
        else:
            self.send_error(404)

def run_server():
    print("=" * 60)
    print(f"🚀 Launching 3D Keypoint Mapping Server on http://localhost:{PORT}")
    print("=" * 60)
    os.chdir(REPO_ROOT)
    server = HTTPServer(("localhost", PORT), MappingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    run_server()
