#!/usr/bin/env python3
import http.server
import socketserver
import threading
import sys

PORTS = []

class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server(port):
    with socketserver.TCPServer(("127.0.0.1", port), HealthHandler) as httpd:
        httpd.serve_forever()

def main():
    ports = [int(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [50051, 50052]
    threads = []
    for p in ports:
        t = threading.Thread(target=run_server, args=(p,), daemon=True)
        t.start()
        threads.append(t)
    # Keep the main thread alive while servers run
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
