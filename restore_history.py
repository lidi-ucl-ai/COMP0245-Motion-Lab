"""Verify and restore the COMP0245 GitHub Release assets."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Archive path leaves destination: {name}')
    return path


def verify(path, info):
    if path.stat().st_size != info['bytes']:
        raise ValueError(f'Wrong file size: {path}')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != info['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')


def restore(assets, dest, subset, verify_only):
    manifest = json.loads((assets / 'archive-manifest.json').read_text())
    archives = [manifest['models']] if subset == 'models' else manifest['history']
    for info in archives:
        verify(safe_path(assets, info['name']), info)
        print(f'Verified {info["name"]}', flush=True)
    if verify_only:
        return
    for info in archives:
        with ZipFile(safe_path(assets, info['name'])) as archive:
            for member in archive.infolist():
                target = safe_path(dest, member.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open('xb') as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
    if subset == 'history':
        for link in manifest['symlinks']:
            if link['status'] == 'archived':
                source = safe_path(dest, link['path'])
                target = safe_path(dest, link['target'])
                if not target.exists():
                    raise FileNotFoundError(target)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=link['directory'])
    print(f'Restored {subset} to {dest}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets', type=Path)
    parser.add_argument('--dest', type=Path, required=True)
    parser.add_argument('--subset', choices=['models', 'history'], required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    restore(args.assets.resolve(), args.dest.resolve(), args.subset, args.verify_only)


if __name__ == '__main__':
    main()
