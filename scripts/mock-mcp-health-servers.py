#!/usr/bin/env python3
import http.server
import socketserver
import threading
import sys

def run_server(port):
    class Handler(http.server.BaseHTTPRequestHandler):
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
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.serve_forever()

def main():
    ports = [50051, 50052] if len(sys.argv) <= 1 else [int(p) for p in sys.argv[1:]]
    threads = []
    for p in ports:
        t = threading.Thread(target=run_server, args=(p,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
