"""Build one course submission ZIP from the reviewed source inventory. No ML imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'docs/delivery/source_files.json'
LIMIT = 50_000_000  # Conservative decimal 50 MB.
BLOCKED_PARTS = {'.git', '.local', 'node_modules', '__pycache__', 'checkpoints', 'weights', '.venv', 'venv'}
BLOCKED_SUFFIXES = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.onnx', '.mp4', '.mkv', '.avi', '.pyc'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive path: ' + name)
    if set(p.parts) & BLOCKED_PARTS or any(s.startswith('.venv') for s in p.parts):
        raise ValueError('Excluded runtime directory: ' + name)
    if p.suffix.lower() in BLOCKED_SUFFIXES or p.name.startswith('.env'):
        raise ValueError('Excluded artifact: ' + name)
    if re.fullmatch(r'IMCS-V2_(train|dev|test)\.json', p.name, re.I):
        raise ValueError('Raw dataset must remain external: ' + name)
    return name


def inspect_nested(data, depth=0):
    if depth > 2:
        raise ValueError('Unexpected deeply nested ZIP')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.testzip():
            raise ValueError('Nested ZIP CRC failed')
        for entry in z.infolist():
            safe_name(entry.filename.rstrip('/'))
            if entry.file_size > LIMIT:
                raise ValueError('Oversized archived member: ' + entry.filename)
            if entry.filename.endswith('.zip'):
                inspect_nested(z.read(entry), depth+1)


def reviewed_files(root=ROOT):
    manifest = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    result = []
    seen = set()
    for row in manifest['files']:
        name = safe_name(row['path'])
        p = (root/name).resolve()
        if name in seen or not p.is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError('Missing, duplicated, or escaped source: ' + name)
        seen.add(name)
        data = p.read_bytes()
        # Git may normalize line endings on a fresh Windows checkout.
        if digest(data) != row['sha256']:
            try:
                normalized = data.decode('utf-8').replace('\r\n', '\n').encode('utf-8')
            except UnicodeDecodeError:
                normalized = b''
            if not row.get('lf_sha256') or digest(normalized) != row['lf_sha256']:
                raise ValueError('Reviewed source changed; update inventory after review: ' + name)
        if p.suffix.lower() == '.zip':
            inspect_nested(data)
        result.append((name, data))
    result.append((MANIFEST, (root/MANIFEST).read_bytes()))
    return result


def build(output, files, *, source_only):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [n for n, _ in files]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate submission filename')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in sorted(files):
            z.writestr(safe_name(name), data)
        z.writestr('交付状态.json', json.dumps({
            'scope': 'source_preview' if source_only else 'submission_with_user_documents',
            'report_and_contributions_included': not source_only,
            'notice': '源码预览：正式报告PDF、每人贡献声明由提交人后补。' if source_only else '请本人检查报告内AI声明、视频链接、贡献比例及签名。',
            'file_sha256': {name: digest(data) for name, data in files}
        }, ensure_ascii=False, indent=2))
    if output.stat().st_size > LIMIT:
        # This file was exclusively created above; no existing file is touched.
        output.unlink()
        raise ValueError('ZIP exceeds 50,000,000 bytes; reduce the added submission documents')
    with zipfile.ZipFile(output) as z:
        if z.testzip():
            raise ValueError('ZIP CRC verification failed')
        count = len(z.namelist())
    return {'path': str(output), 'bytes': output.stat().st_size,
            'sha256': digest(output.read_bytes()), 'zip_crc': 'PASS',
            'files': count, 'limit_bytes': LIMIT, 'source_only': source_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check reviewed source identity only; no prediction or training')
    parser.add_argument('--source-only', action='store_true', help='Create source preview without final report and declarations')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--leader-id')
    parser.add_argument('--leader-name')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--contribution', type=Path, action='append', default=[])
    args = parser.parse_args()
    files = reviewed_files()
    if args.check:
        print(json.dumps({'status': 'PASS', 'reviewed_files': len(files), 'training': False, 'inference': False}, ensure_ascii=False))
        return
    if not args.output:
        parser.error('--output is required')
    if args.source_only:
        if args.report or args.contribution:
            parser.error('Do not mix --source-only with final submission documents')
    else:
        if not all([args.leader_id, args.leader_name, args.report, args.contribution]):
            parser.error('Final ZIP requires --leader-id --leader-name --report and each --contribution')
        expected = f'{args.leader_id}_{args.leader_name}_NLPExp.zip'
        if args.output.name != expected:
            parser.error('Required ZIP filename: ' + expected)
        report_data = args.report.read_bytes()
        if args.report.suffix.lower() != '.pdf' or not report_data.startswith(b'%PDF-'):
            parser.error('--report must be a PDF')
        files.append(('报告.pdf', report_data))
        for p in args.contribution:
            if not re.fullmatch(r'贡献声明_[^_]+_[^_]+\.(md|pdf|docx)', p.name):
                parser.error('Name each contribution: 贡献声明_学号_姓名.md/pdf/docx')
            data = p.read_bytes()
            if p.suffix == '.docx':
                inspect_nested(data)
            files.append((p.name, data))
    print(json.dumps(build(args.output, files, source_only=args.source_only), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
