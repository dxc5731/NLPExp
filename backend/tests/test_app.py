import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app import app
from backend.entities import entities, strict_score, differences
from backend.common import verify_assets, config

class StrictEntitiesTests(unittest.TestCase):
    def test_boundary_overlap_is_not_tp(self):
        gold=['O']*5+['B-Symptom','I-Symptom','I-Symptom']+['O']*2
        pred=['O']*5+['B-Symptom']+['I-Symptom']*4
        result=strict_score(gold,pred)
        self.assertEqual((result['tp'],result['fp'],result['fn'],result['f1']),(0,1,1,0))
    def test_unicode_coordinates_and_exact_text(self):
        e=entities('😀 发热。',['O','O','B-Symptom','I-Symptom','O'])[0]
        self.assertEqual((e['start'],e['end'],e['text']),(2,4,'发热'))
    def test_duplicate_text_different_positions(self):
        g=['B-Symptom','I-Symptom','O','O'];p=['O','O','B-Symptom','I-Symptom']
        self.assertEqual(strict_score(g,p)['tp'],0)
    def test_illegal_bio_rejected(self):
        with self.assertRaises(ValueError):entities('烧',['I-Symptom'])
    def test_no_entities_and_all_o(self):self.assertEqual(entities('天气好',['O']*3),[])
    def test_wrong_type_not_match(self):
        self.assertEqual(strict_score(['B-Drug'],['B-Drug_Category'])['tp'],0)
    def test_difference_pairing_uses_spans(self):
        results=[{'method':'rule','status':'success','entities':entities('发热发热',['B-Symptom','I-Symptom','O','O'])},
                 {'method':'softmax','status':'success','entities':entities('发热发热',['O','O','B-Symptom','I-Symptom'])}]
        self.assertEqual(len(differences(results)['method_only']),2)
        self.assertEqual(differences(results)['shared_all'],[])

class ApiSafetyTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app)
    def test_post_run_audit_retained_and_registered(self):
        import json,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory(prefix='nlpexp_demo_api_test_') as tmp:
            run_id='dev_demo_20990101_000000_01234567';directory=Path(tmp)/run_id;directory.mkdir()
            (directory/'manifest.json').write_text(json.dumps({'run_id':run_id,'methods':{},'artifacts':{}}),encoding='utf-8')
            audit={'status':'NEEDS_REVIEW','issue':'explicit synthetic API test'}
            (directory/'post_run_audit.json').write_text(json.dumps(audit),encoding='utf-8')
            c=config();c['demo_runs']=tmp
            with patch('backend.app.config',return_value=c):
                self.assertEqual(self.client.get('/api/runs/'+run_id).json()['post_run_audit'],audit)
                self.assertEqual(self.client.get('/api/runs/'+run_id+'/artifacts/post_run_audit').json(),audit)
    def test_health_and_no_full_dev_http(self):
        self.assertFalse(self.client.get('/api/health').json()['full_dev_http_endpoint'])
        self.assertEqual(self.client.post('/api/runs').status_code,405)
    def test_empty_and_long_rejected(self):
        for text in ['', '   ', '字'*501]:self.assertEqual(self.client.post('/api/text/infer',json={'text':text}).status_code,422)
    def test_strict_text_type(self):
        for body in [{'text':2},{'text':'发热','command':'anything'}]:self.assertEqual(self.client.post('/api/text/infer',json=body).status_code,422)
    def test_cross_origin_write_rejected(self):
        self.assertEqual(self.client.post('/api/text/infer',headers={'Origin':'https://example.com'},json={'text':'发热'}).status_code,403)
    def test_arbitrary_run_and_file_rejected(self):
        self.assertEqual(self.client.get('/api/runs/not-a-run').status_code,404)
        self.assertEqual(self.client.get('/api/files?path=C:/Windows/win.ini').status_code,404)
    def test_missing_resource_never_falls_back(self):
        c=config();c['softmax_checkpoint']='C:/missing_nlpexp_resource/best.pt'
        with patch('backend.common.config',return_value=c):
            with self.assertRaises(FileNotFoundError):verify_assets('softmax')
    def test_case_verified_on_original_only(self):
        from backend.text_service import verified_case
        c=verified_case()
        self.assertNotEqual(c['raw_text'],'今天一直是38度左右')
        self.assertEqual(c['gold_entities'][0]['text'],'38度')
        self.assertEqual(c['historical']['crf']['entities'][0]['text'],'38度左右')

if __name__=='__main__':unittest.main()
