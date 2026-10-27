import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.package_delivery import build, digest, inspect_nested, reviewed_files, safe_name, MANIFEST


class PackageSafetyTests(unittest.TestCase):
    def test_private_and_escaping_paths_rejected(self):
        for name in ['../secret', '/root/data', 'C:/data', '.local/input.json',
                     'model/best.pt', 'a/pytorch_model.bin', '.venv-x/python.exe',
                     '.env', 'data/IMCS-V2_dev.json', 'movie.mp4']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)

    def test_nested_weights_rejected(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:z.writestr('weights/best.pt',b'not a real model')
        with self.assertRaises(ValueError):inspect_nested(b.getvalue())

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'existing.zip';p.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):build(p,[('README.md',b'new')],source_only=True)
            self.assertEqual(p.read_bytes(),b'existing')

    def test_archive_has_hash_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'preview.zip';result=build(p,[('README.md',b'example')],source_only=True)
            self.assertEqual(result['zip_crc'],'PASS')
            with zipfile.ZipFile(p) as z:
                info=json.loads(z.read('交付状态.json'))
                self.assertFalse(info['report_and_contributions_included'])
                self.assertEqual(info['file_sha256']['README.md'],digest(b'example'))

    def test_size_limit_rejects_new_archive(self):
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.package_delivery.LIMIT',10):
            p=Path(tmp)/'large.zip'
            with self.assertRaises(ValueError):build(p,[('README.md',b'x')],source_only=True)
            self.assertFalse(p.exists())

    def test_inventory_allows_only_line_ending_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=root/MANIFEST;m.parent.mkdir(parents=True)
            m.write_text(json.dumps({'files':[{'path':'README.md','sha256':digest(b'a\nb\n'),
                'lf_sha256':digest(b'a\nb\n')}]}),encoding='utf-8')
            p=root/'README.md';p.write_bytes(b'a\r\nb\r\n')
            self.assertEqual(len(reviewed_files(root)),2)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):reviewed_files(root)


if __name__=='__main__':unittest.main()
