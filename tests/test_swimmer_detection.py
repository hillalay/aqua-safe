import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from src.swimmer_detection import (
    duplicate_candidates, export_reviewed, source_groups, to_yolo, validate_yolo, write_json,
)
from src.video_review.inventory import sha256_file


class SwimmerDetectionTests(unittest.TestCase):
    def test_box_format_and_bad_labels(self):
        self.assertEqual(to_yolo([[10,20,50,60]],100,100), '0 0.30000000 0.40000000 0.40000000 0.40000000\n')
        self.assertEqual(validate_yolo(''),0)
        self.assertEqual(validate_yolo(to_yolo([[0,0,100,100]],100,100)),1)
        for text in ['1 .5 .5 .2 .2','0 nan .5 .2 .2','0 .95 .5 .2 .2','0 .5 .5 0 .2','0 .5 .5 .2 .2 .9']:
            with self.subTest(text=text),self.assertRaises(ValueError):validate_yolo(text)
        with self.assertRaises(ValueError):to_yolo([[-1,0,10,10]],100,100)

    def test_duplicate_geometry_is_only_a_candidate_and_groups_are_transitive(self):
        self.assertEqual(duplicate_candidates([[0,0,100,100],[10,10,30,30],[200,0,300,100]]),[[0,1]])
        inventory=[{'sha256':'A','source_group':'one'},{'sha256':'B','source_group':'one'},
                   {'sha256':'C','source_group':'two'}]
        clips=[{'sha256':'B','leakage_group':'linked'},{'sha256':'C','leakage_group':'linked'}]
        groups=source_groups(inventory,clips)
        self.assertEqual(len(set(groups.values())),1)

    def package(self,root):
        for name in ['images','review/labels']:(root/name).mkdir(parents=True)
        manifest=[];reviews={};groups={}
        for key,split,color in [('a','train','white'),('b','validation','black')]:
            image=root/'images'/f'{key}.png';Image.new('RGB',(100,100),color).save(image)
            manifest.append({'image_id':key,'source_group':key,'held_out':False,'image_sha256':sha256_file(image)})
            reviews[key]={'status':'verified','reviewer_type':'human','reviewer':'Tester','all_people_reviewed':True}
            groups[key]={'human_confirmed':True,'reviewer':'Tester','capture_group':key,'split':split}
            (root/'review/labels'/f'{key}.txt').write_text('0 .5 .5 .2 .2\n')
        write_json(root/'manifest.json',manifest);write_json(root/'reviews.json',reviews)
        write_json(root/'group_review.json',groups)
        write_json(root/'preparation_report.json',{'manifest_sha256':sha256_file(root/'manifest.json')})
        return reviews,groups

    def test_export_only_verified_labels_and_correct_yaml_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package=root/'package';self.package(package);out=root/'export'
            export_reviewed(package,out)
            config=yaml.safe_load((out/'data.yaml').read_text())
            self.assertEqual(config['names'],{0:'swimmer'})
            self.assertTrue((out/config['val']).is_dir())
            self.assertEqual(len(list((out/'labels').rglob('*.txt'))),2)
            with self.assertRaises(FileExistsError):export_reviewed(package,out)

    def test_unreviewed_labels_missing_labels_and_group_leakage_block_export(self):
        for failure in ['automatic','missing','leakage','holdout','changed_manifest']:
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);package=root/'package';reviews,groups=self.package(package)
                if failure=='automatic':
                    reviews['a']['reviewer_type']='assistant';(package/'reviews.json').write_text(json.dumps(reviews))
                elif failure=='missing':(package/'review/labels/a.txt').unlink()
                elif failure=='leakage':
                    groups['b']['capture_group']='a';(package/'group_review.json').write_text(json.dumps(groups))
                elif failure=='holdout':
                    m=json.loads((package/'manifest.json').read_text());m[0]['held_out']=True
                    (package/'manifest.json').write_text(json.dumps(m))
                    (package/'preparation_report.json').write_text(json.dumps({'manifest_sha256':sha256_file(package/'manifest.json')}))
                else:(package/'manifest.json').write_text('[]')
                with self.assertRaises((ValueError,FileNotFoundError)):export_reviewed(package,root/'export')
                self.assertFalse((root/'export').exists())


if __name__=='__main__':unittest.main()
