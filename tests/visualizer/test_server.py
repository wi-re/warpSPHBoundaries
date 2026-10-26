"""Test HTTP contracts without requiring socket permissions in CI/sandboxes."""
from io import BytesIO
import json
import subprocess
import sys
import pytest

from warpSPHBoundaries.visualizer.__main__ import Handler, STATIC


def request(path, body=b'', origin=None):
    # Exercise the actual request handlers with in-memory transport streams.
    handler = object.__new__(Handler)
    handler.path = path
    handler.headers = {'Content-Length': str(len(body)), 'Host': '127.0.0.1:8765'}
    if origin:
        handler.headers['Origin'] = origin
    handler.rfile, handler.wfile = BytesIO(body), BytesIO()
    handler.send_response = lambda status: setattr(handler, 'status', status)
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    return handler


def test_info_and_compute_contract():
    h = request('/api/info')
    h.do_GET()
    assert h.status == 200
    assert 'mesh' in json.loads(h.wfile.getvalue())['models']
    h = request('/api/compute', json.dumps({'resolution': 17}).encode(), 'http://127.0.0.1:8765')
    h.do_POST()
    assert h.status == 200
    assert json.loads(h.wfile.getvalue())['schema_version'] == 1


def test_http_validation_and_origin_check():
    for payload in (b'invalid', b'{"resolution": 1000}', b'[]', b''):
        h = request('/api/compute', payload)
        h.do_POST()
        assert h.status == 400
        assert 'error' in json.loads(h.wfile.getvalue())
    h = request('/api/compute', b'{}', 'https://another-site.invalid')
    h.do_POST()
    assert h.status == 403
    h = request('/api/unknown', b'{}')
    h.do_POST()
    assert h.status == 404


def test_3d_http_response_is_a_bounded_encoded_volume():
    h = request('/api/compute', json.dumps({'dimension': 3, 'resolution': 17}).encode())
    h.do_POST()
    assert h.status == 200
    result = json.loads(h.wfile.getvalue())
    assert result['dimension'] == 3 and result['shape'] == [17]*3
    assert result['encoding'] == 'base64-f32-le'


@pytest.mark.parametrize('dimension', [2, 3])
def test_static_export_contains_viewer_and_reopenable_snapshot(tmp_path, dimension):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'config': {'resolution': 17, 'dimension': dimension},
                                 'view': {'colormap': 'magma', 'sliceOffsets': [.1, .2, .3]}}))
    output = tmp_path/'site'
    subprocess.run([sys.executable, '-m', 'warpSPHBoundaries.visualizer',
                    '--export', str(output), '--config', str(config)], check=True, capture_output=True)
    assert {p.name for p in STATIC.iterdir()} <= {p.name for p in output.iterdir()}
    result = json.loads((output/'example.json').read_text())
    assert result['dimension'] == dimension and result['config']['resolution'] == 17
    if dimension == 2:
        assert len(result['results'][0]['values']) == 17**2
    else:
        assert result['shape'] == [17]*3
        assert result['view']['colormap'] == 'magma'
        assert (output/'volume.wgsl').is_file() and (output/'volume.html').is_file()
    assert './app.js' in (output/'index.html').read_text()
