"""Export browser-only Boundary Laboratory assets without importing Warp.

Serve the directory over HTTPS (GitHub Pages) or localhost. No compute service,
frontend build, CDN or runtime package installation is required.
"""
import argparse
from pathlib import Path
import shutil


def export_static(destination):
    source=Path(__file__).resolve().parents[2]/'src/warpSPHBoundaries/visualizer/static'
    destination=Path(destination).resolve()
    if destination==source or source in destination.parents or destination in source.parents:
        raise ValueError('Export to a separate directory outside the source tree')
    destination.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.is_file():shutil.copy2(path,destination/path.name)
    (destination/'.nojekyll').touch()
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(export_static(args.destination))
