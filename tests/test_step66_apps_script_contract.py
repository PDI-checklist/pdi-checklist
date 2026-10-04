import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from central_write_adapter import HTTPJSONCentralWriteAdapter, WriteStatus


class _Handler(BaseHTTPRequestHandler):
    captured = []
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = json.loads(self.rfile.read(length).decode('utf-8'))
        self.__class__.captured.append(body)
        response = {"status": "SUCCESS", "message": "Observation batch committed and confirmed.", "records_written": len(body.get("observations", []))}
        raw = json.dumps(response).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def log_message(self, *_args):
        pass


def test_apps_script_adapter_sends_token_and_metadata():
    server = HTTPServer(('127.0.0.1', 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        adapter = HTTPJSONCentralWriteAdapter(
            base_url=f'http://127.0.0.1:{server.server_port}/exec',
            token='test-secret',
        )
        adapter.apps_script = True
        result = adapter.write_batch('VH-147', 'S. Harish', [{
            'JPC Number': 'VH147', 'Inspector Name': 'S. Harish',
            'Observation': 'Door bush missing', 'Department': 'Integration',
            'Station': 'PDI Stage', 'Defect Category': 'Missing Part',
            'Status': 'Closed', 'Cleared by': 'S. Harish',
            'Closure date': '01-10-2026', 'Closure Remarks': 'Verified OK',
            'OCR Status': 'OK', 'OCR Confidence': 0.96,
        }])
        assert result.status is WriteStatus.SUCCESS
        body = _Handler.captured[-1]
        assert body['token'] == 'test-secret'
        assert body['action'] == 'write_batch'
        assert body['jpc'] == 'VH-147'
        assert body['observations'][0]['Department'] == 'Integration'
        assert body['observations'][0]['Closure date'] == '01-10-2026'
    finally:
        server.shutdown()
        thread.join(timeout=2)
