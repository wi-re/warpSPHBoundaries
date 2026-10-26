"""Dependency-light, loopback-only HTTP service for the boundary laboratory."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import urlsplit

from .compute import DEFAULT, FIELDS, KERNELS, MODELS, compute
from .compute3d import DEFAULT3D, MODELS3D
from .terms3d import FIELDS3D

STATIC = Path(__file__).with_name("static")
COMPUTE_LOCK = threading.Lock()  # bound memory; Torch/table initialization is shared


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def json(self, status, data):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/info":
            return self.json(200, dict(default=DEFAULT, models=MODELS, kernels=KERNELS, fields=FIELDS,
                                       default3d=DEFAULT3D, models3d=MODELS3D, fields3d=FIELDS3D))
        if path.startswith("/api/"):
            return self.json(404, {"error": "Unknown API route"})
        super().do_GET()

    def do_POST(self):
        route = urlsplit(self.path).path
        if route not in ("/api/compute", "/api/wall", "/api/wall/compare"):
            return self.json(404, {"error": "Unknown API route"})
        # Same-origin only: static Pages can consume exported results without exposing
        # local compute to arbitrary websites. A future remote provider is explicit.
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.json(403, {"error": "Use the locally served interface for live computation"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 32768:
                raise ValueError("Request must contain at most 32 KiB of JSON")
            payload = json.loads(self.rfile.read(length))
            with COMPUTE_LOCK:
                if route in ("/api/wall", "/api/wall/compare"):
                    from .wall import compute_wall, compare_wall
                    result = (compare_wall if route.endswith('/compare') else compute_wall)(payload)
                else:
                    result = compute(payload)
            self.json(200, result)
        except (ValueError, TypeError) as exc:
            self.json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            self.json(500, {"error": "Evaluation failed. See the Python server log for details."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--export", type=Path, metavar="DIRECTORY", help="Export static viewer and computed example.json; no server")
    parser.add_argument("--config", type=Path, help="JSON configuration or an exported result whose config should be used")
    parser.add_argument("--dimension", type=int, choices=(2, 3), help="Dimension of the exported example (with --export)")
    parser.add_argument("--wall", action="store_true", help="Export the native mixed-wall diagnostic (with --export)")
    parser.add_argument("--compare", action="store_true", help="Include exhaustive/refinement/precision comparisons with --wall --export")
    args = parser.parse_args()
    if args.compare and not (args.wall and args.export):
        parser.error('--compare requires --wall --export')
    if args.export:
        config = json.loads(args.config.read_text()) if args.config else {}
        view = config.get('view') if isinstance(config, dict) else None
        if isinstance(config, dict) and "config" in config:
            config = config["config"]
        if args.dimension is not None:
            config = {**config, 'dimension': args.dimension}
        if args.wall:
            from .wall import compute_wall, compare_wall
            result = (compare_wall if args.compare else compute_wall)(config)
        else:
            result = compute(config)
        if result['dimension'] == 3 and isinstance(view, dict):
            result['view'] = view
        destination = args.export.resolve()
        if destination == STATIC.resolve() or STATIC.resolve() in destination.parents:
            parser.error("Export to a separate directory")
        destination.mkdir(parents=True, exist_ok=True)
        for source in STATIC.iterdir():
            if source.is_file():
                shutil.copy2(source, destination/source.name)
        # Keep the original viewers' example contract independent of the wall
        # diagnostic's 2D slice payload, including for a 3D wall export.
        example = compute({"resolution":17, "kernel":config.get("kernel","w2")}) if args.wall else result
        (destination/"example.json").write_text(json.dumps(example, allow_nan=False, separators=(",", ":")))
        if args.wall:
            (destination/"wall-example.json").write_text(json.dumps(result, allow_nan=False, separators=(",", ":")))
        print(f"Static viewer exported to {destination}; serve it with python -m http.server or GitHub Pages")
        return
    if args.config or args.dimension or args.wall:
        parser.error("--config and --dimension are used with --export; choose the live dimension in the viewer")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Boundary laboratory: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
