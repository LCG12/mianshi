"""下载锁定版本的三个模型；支持断点续传，逐文件检查 SHA256。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import urlencode


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def download_file(model, entry, root):
    folder = root / model['repo'].split('/')[-1]
    target = folder / entry['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) == entry['sha256']:
        print('VERIFIED', target, flush=True)
        return
    partial = target.with_name(target.name + '.part')
    query = urlencode({'Revision': model['revision'], 'FilePath': entry['path']})
    url = f"https://modelscope.cn/api/v1/models/{model['repo']}/repo?{query}"
    print('DOWNLOADING', target, entry['size'], flush=True)
    if shutil.which('aria2c'):
        command = ['aria2c', '-x', '8', '-s', '8', '-k', '16M', '--continue=true', '--async-dns=false',
                   '--file-allocation=none', '--auto-file-renaming=false',
                   '--summary-interval=0', '--console-log-level=warn',
                   '--download-result=hide', '--max-tries=5', '--retry-wait=3',
                   '-d', str(partial.parent), '-o', partial.name, url]
    else:
        command = ['curl', '--fail', '--location', '--retry', '3',
                   '--connect-timeout', '20', '--max-time', '1800',
                   '--silent', '--show-error', '--continue-at', '-',
                   '--output', str(partial), url]
    subprocess.run(command, check=True)
    if partial.stat().st_size != entry['size'] or digest(partial) != entry['sha256']:
        raise RuntimeError(f'文件校验失败：{target}')
    partial.replace(target)
    print('VERIFIED', target, flush=True)


def main():
    manifest = json.loads(Path(__file__).with_name('models.lock.json').read_text())
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('models')
    files = [(model, entry) for model in manifest for entry in model['files']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: download_file(*item, root), files))
    print('ALL_MODELS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
