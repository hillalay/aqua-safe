"""Detection pilot and human-reviewed swimmer dataset preparation; never trains."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import yaml

from src.drowning_demo.analysis import ROOT, decoded_frames, overlay_font
from src.drowning_demo.detection_review import SOURCES
from src.person_dataset import intersection
from src.video_classification.binary_drowning import Groups, read_csv
from src.video_review.inventory import sha256_file, difference_hash, hamming


OUTPUT = ROOT / 'results/swimmer_detection_pilot_v1'


def write_json(path, value):
    with path.open('x', encoding='utf-8') as target:
        json.dump(value, target, ensure_ascii=False, indent=2)


def duplicate_candidates(boxes):
    pairs = []
    for i, j in itertools.combinations(range(len(boxes)), 2):
        a, b = boxes[i], boxes[j]
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        shared = intersection(a, b)
        if shared / max(area_a+area_b-shared, 1) >= .5 or shared / max(min(area_a, area_b), 1) >= .8:
            pairs.append([i, j])
    return pairs


def pilot(output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'frames').mkdir()
    (output/'comparisons').mkdir()
    frames = []
    for source, anchors in SOURCES.items():
        for index, frame in enumerate(decoded_frames(ROOT/source)):
            if index not in anchors:
                continue
            image = frame.to_ndarray(format='bgr24')
            image_id = Path(source).stem + f'_F{index:05d}'
            Image.fromarray(image[:, :, ::-1]).save(output/'frames'/f'{image_id}.png')
            frames.append((source, index, image_id, image))
    assert len(frames) == sum(map(len, SOURCES.values()))
    model = YOLO(str(ROOT/'yolov8s.pt'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report = {'sources': {p: sha256_file(ROOT/p) for p in SOURCES},
              'weights_sha256': sha256_file(ROOT/'yolov8s.pt'), 'device': device,
              'model_names': model.names, 'model_task': model.task,
              'checkpoint_dataset': model.ckpt.get('train_args', {}).get('data'),
              'frames_per_configuration': len(frames), 'repeats': 3, 'experiments': [],
              'manual_ground_truth_status': 'absent; human review required',
              'scope': 'Same 36 raw frames, batch=1, predecoded inputs; excludes video decode/render. Not an independent accuracy benchmark.'}
    for confidence, size in itertools.product((.10, .05), (960, 1280)):
        if device == 'cuda':
            torch.cuda.empty_cache()
        for _ in range(3):
            model.predict(frames[0][3], classes=[0], conf=confidence, imgsz=size,
                          iou=.7, device=device, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        records, durations = [], []
        for repeat in range(3):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            for source, index, image_id, image in frames:
                result = model.predict(image, classes=[0], conf=confidence, imgsz=size,
                                       iou=.7, device=device, verbose=False)[0]
                boxes, scores = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
                if repeat == 0:
                    records.append({'source': source, 'frame': index, 'image_id': image_id,
                                    'boxes': boxes, 'confidence': scores,
                                    'duplicate_candidate_pairs': duplicate_candidates(boxes),
                                    'human_verified_missed_people': None,
                                    'human_verified_duplicate_boxes': None})
            if device == 'cuda':
                torch.cuda.synchronize()
            durations.append(time.perf_counter()-start)
        experiment = {'confidence': confidence, 'imgsz': size, 'nms_iou': .7,
            'boxes': sum(len(r['boxes']) for r in records),
            'frames_with_boxes': sum(bool(r['boxes']) for r in records),
            'duplicate_candidate_pairs': sum(len(r['duplicate_candidate_pairs']) for r in records),
            'human_verified_missed_people': None, 'human_verified_duplicate_boxes': None,
            'fps': len(frames)*3/sum(durations), 'repeat_seconds': durations,
            'peak_torch_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
            'peak_torch_reserved_mib': torch.cuda.max_memory_reserved()/2**20 if device == 'cuda' else None,
            'predictions': records}
        report['experiments'].append(experiment)
        print({k:v for k,v in experiment.items() if k != 'predictions'}, flush=True)
    write_json(output/'pilot_results.json', report)
    font = overlay_font(18)
    for source, index, image_id, bgr in frames:
        sheet = Image.new('RGB', (1280, 800), '#101923')
        for col, exp in enumerate(report['experiments']):
            record = next(r for r in exp['predictions'] if r['image_id'] == image_id)
            image = Image.fromarray(bgr[:, :, ::-1]).resize((640, 360))
            draw = ImageDraw.Draw(image)
            for bi, (box, score) in enumerate(zip(record['boxes'], record['confidence'])):
                xy = [box[0]*640/bgr.shape[1], box[1]*360/bgr.shape[0],
                      box[2]*640/bgr.shape[1], box[3]*360/bgr.shape[0]]
                draw.rectangle(xy, outline='#30ef90', width=2)
                draw.text((xy[0], max(0, xy[1]-20)), f'{bi}: {score:.2f}', font=font, fill='#ffc857',
                          stroke_width=1, stroke_fill='black')
            x, y = (col % 2)*640, (col // 2)*400
            sheet.paste(image, (x, y+35))
            ImageDraw.Draw(sheet).text((x+5, y+5), f"conf={exp['confidence']} size={exp['imgsz']} boxes={len(record['boxes'])}", font=font, fill='white')
        sheet.save(output/'comparisons'/f'{image_id}.jpg')
    reviews = [{'image_id': image_id, 'source': source, 'frame': index, 'status': 'pending',
                'reviewer': '', 'reviewer_type': '', 'boxes_xyxy': [], 'all_people_reviewed': False,
                'notes': ''} for source, index, image_id, _ in frames]
    write_json(output/'human_reference_review.json', reviews)
    assert report['weights_sha256'] == sha256_file(ROOT/'yolov8s.pt')
    assert all(sha256_file(ROOT/p) == digest for p,digest in report['sources'].items())
    return report


def to_yolo(boxes, width, height):
    lines = []
    for x1, y1, x2, y2 in boxes:
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError('Invalid pixel box')
        lines.append(f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return '\n'.join(lines) + ('\n' if lines else '')


def validate_yolo(text):
    count = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != '0':
            raise ValueError('Expected swimmer class 0 and four coordinates')
        x,y,w,h = map(float, fields[1:])
        if (not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0
                or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6):
            raise ValueError('Invalid normalized box')
        count += 1
    return count


def source_groups(inventory, old_clips):
    groups = Groups([r['sha256'] for r in inventory])
    known = {}
    for rows, key in ((inventory,'source_group'), (old_clips,'leakage_group')):
        for row in rows:
            digest = row['sha256']
            if digest not in groups.parent:
                continue
            group = row[key]
            if not group or group.lower() == 'unknown':
                continue
            token = (key,group)
            if token in known:
                groups.union(known[token],digest)
            else:
                known[token] = digest
    return {digest: groups.find(digest) for digest in groups.parent}


def prepare(output, count=400):
    if not 300 <= count <= 500:
        raise ValueError('Pilot requires 300-500 images')
    inventory = read_csv(ROOT/'results/person_dataset_new_videos/inventory.csv')
    old_clips = read_csv(ROOT/'results/aquasafe_yolo_r3d18_v1/clip_manifest.csv')
    groups = source_groups(inventory, old_clips)
    heldout = {groups[r['sha256']] for r in old_clips if r['draft_split']=='validation'}
    heldout.update(groups[next(r['sha256'] for r in inventory if (ROOT/r['relative_path']).resolve()==(ROOT/p).resolve())] for p in SOURCES)
    unique = {}
    for row in inventory:
        unique.setdefault(row['sha256'], row)
    multiple = {str((ROOT/r['path']).resolve()).lower() for r in read_csv(ROOT/'results/video_review/same_style_multiple_swimmers.csv')}
    strata = defaultdict(list)
    for row in unique.values():
        path = ROOT/row['relative_path']
        stratum = 'multi_candidate' if str(path.resolve()).lower() in multiple else row['source_category_hint']
        strata[stratum].append(row)
    targets = {'DROWNING': round(count*.4), 'multi_candidate': round(count*.15)}
    targets['NORMAL'] = count - sum(targets.values())
    selected = []
    for stratum, target in targets.items():
        candidates = []
        for row in sorted(strata[stratum], key=lambda r:r['sha256']):
            # At most eight temporally spaced images per source avoids dense near-duplicates.
            n = min(8, max(1, int(float(row['duration_sec']))))
            indices = [round((int(row['frame_count'])-1)*(i+.5)/n) for i in range(n)]
            candidates.append([(row,index,stratum) for index in indices])
        pool = [item for level in itertools.zip_longest(*candidates) for item in level if item is not None]
        if len(pool) < target:
            raise ValueError(f'Insufficient distinct source frames for {stratum}: {len(pool)} < {target}')
        selected.extend(pool[:target])
    output.mkdir(parents=True, exist_ok=False)
    for folder in ['images','draft_labels','review/labels']:
        (output/folder).mkdir(parents=True)
    model = YOLO(str(ROOT/'yolov8s.pt'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    manifest, reviews, group_review = [], {}, {}
    by_source = defaultdict(list)
    for row,index,stratum in selected:
        by_source[row['sha256']].append((row,index,stratum))
    for digest, picks in by_source.items():
        row = picks[0][0]; source = ROOT/row['relative_path']
        if sha256_file(source) != digest:
            raise ValueError('Source differs from inventory: '+str(source))
        cap = cv2.VideoCapture(str(source))
        try:
            for row,index,stratum in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image = cap.read()
                if not ok or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1)) > .5:
                    raise ValueError('Frame read failed: '+str(source))
                image_id = digest[:20]+f'_F{index:07d}'
                destination = output/'images'/f'{image_id}.png'
                if destination.exists():
                    raise ValueError('Duplicate selected frame')
                Image.fromarray(image[:,:,::-1]).save(destination)
                result = model.predict(image, classes=[0], conf=.05, imgsz=1280, device=device, verbose=False)[0]
                text = to_yolo(result.boxes.xyxy.cpu().tolist(),image.shape[1],image.shape[0])
                (output/'draft_labels'/f'{image_id}.txt').write_text(text,encoding='utf-8')
                group = groups[digest]
                split = 'test' if group in heldout else ('validation' if int(group[:8],16)%5==0 else 'train')
                group_review[group] = {'capture_group':group, 'split':split, 'human_confirmed':False,
                                       'reviewer':'', 'held_out':group in heldout}
                manifest.append({'image_id':image_id,'source_path':str(source.relative_to(ROOT)),
                    'source_sha256':digest,'source_group':group,'frame':index,'width':image.shape[1],
                    'height':image.shape[0],'stratum_hint':stratum,'image_sha256':sha256_file(destination),
                    'held_out':group in heldout,
                    'draft_boxes':validate_yolo(text),'visual_hash':difference_hash(image)})
                reviews[image_id] = {'status':'pending','reviewer':'','reviewer_type':'',
                    'all_people_reviewed':False,'tags':[],'notes':''}
        finally:
            cap.release()
        if sha256_file(source) != digest:
            raise ValueError('Source changed during preparation')
        print(f'Prepared {len(manifest)}/{count} images',flush=True)
    assert len(manifest)==count and len({r['image_id'] for r in manifest})==count
    write_json(output/'manifest.json',manifest)
    write_json(output/'reviews.json',reviews)
    write_json(output/'group_review.json',group_review)
    config={'path':str((output/'exported_dataset').resolve()),'train':'images/train','val':'images/validation',
            'test':'images/test','names':{0:'swimmer'}}
    with (output/'data.template.yaml').open('x',encoding='utf-8') as target:
        yaml.safe_dump(config,target,allow_unicode=True,sort_keys=False)
    write_json(output/'preparation_report.json',{'images':count,'sources':len(by_source),
        'strata':dict(Counter(r['stratum_hint'] for r in manifest)),
        'draft_splits':dict(Counter(group_review[r['source_group']]['split'] for r in manifest)),
        'draft_boxes':sum(r['draft_boxes'] for r in manifest),'human_verified_images':0,
        'missing_human_label_files':count,'capture_groups_confirmed':False,
        'proposal_model':'yolov8s.pt conf=.05 imgsz=1280; automatic proposals, not ground truth',
        'near_duplicate_candidate_pairs':sum(hamming(a['visual_hash'],b['visual_hash'])<=2
            for a,b in itertools.combinations(manifest,2)),
        'manifest_sha256':sha256_file(output/'manifest.json'),'training_started':False})


def export_reviewed(package, destination):
    preparation=json.loads((package/'preparation_report.json').read_text(encoding='utf-8'))
    if sha256_file(package/'manifest.json')!=preparation['manifest_sha256']:
        raise ValueError('Preparation manifest changed')
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    reviews=json.loads((package/'reviews.json').read_text(encoding='utf-8'))
    groups=json.loads((package/'group_review.json').read_text(encoding='utf-8'))
    if len({r['image_id'] for r in manifest})!=len(manifest):
        raise ValueError('Duplicate image IDs')
    checked=[]; splits=defaultdict(set); positives=Counter(); image_splits=defaultdict(set)
    for row in manifest:
        key=row['image_id']
        if not key or Path(key).name!=key or '/' in key or '\\' in key:
            raise ValueError('Invalid image ID')
        review=reviews[key];group=groups[row['source_group']]
        if (review['status']!='verified' or review['reviewer_type']!='human' or not review['reviewer'].strip()
                or review['all_people_reviewed'] is not True):
            raise ValueError('Human box review required: '+key)
        if group['human_confirmed'] is not True or not group['reviewer'].strip() or not group['capture_group'].strip():
            raise ValueError('Human capture-group review required')
        split=group['split']
        if split not in {'train','validation','test'} or (row['held_out'] and split!='test'):
            raise ValueError('Invalid split or changed held-out group')
        splits[group['capture_group']].add(split)
        image=package/'images'/f'{key}.png'; label=package/'review/labels'/f'{key}.txt'
        if sha256_file(image)!=row['image_sha256']:
            raise ValueError('Image changed: '+key)
        # An explicitly reviewed empty file is a negative; a missing file is never a negative.
        text=label.read_text(encoding='utf-8');positives[split]+=validate_yolo(text)
        image_splits[row['image_sha256']].add(split)
        checked.append((image,text,split))
    if any(len(values)>1 for values in [*splits.values(),*image_splits.values()]):
        raise ValueError('Capture group crosses dataset splits')
    if not {'train','validation'} <= {r[2] for r in checked}:
        raise ValueError('Need independent train and validation groups')
    if not positives['train'] or not positives['validation']:
        raise ValueError('Train and validation each need reviewed swimmer boxes')
    destination.mkdir(parents=True,exist_ok=False)
    for split in {r[2] for r in checked}:
        (destination/'images'/split).mkdir(parents=True)
        (destination/'labels'/split).mkdir(parents=True)
    for image,text,split in checked:
        shutil.copyfile(image,destination/'images'/split/image.name)
        (destination/'labels'/split/(image.stem+'.txt')).write_text(text,encoding='utf-8')
    data={'path':str(destination.resolve()),'train':'images/train','val':'images/validation','names':{0:'swimmer'}}
    if 'test' in {r[2] for r in checked}:data['test']='images/test'
    with (destination/'data.yaml').open('x',encoding='utf-8') as target:yaml.safe_dump(data,target,sort_keys=False)
    write_json(destination/'export_report.json',{'images':len(checked),'human_verified':True,'training_started':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pilot','prepare','export'])
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--package',type=Path,default=OUTPUT/'annotation_package')
    parser.add_argument('--count',type=int,default=400)
    args=parser.parse_args()
    if args.command=='pilot':pilot(args.output.resolve())
    elif args.command=='prepare':prepare(args.package.resolve(),args.count)
    else:export_reviewed(args.package.resolve(),args.output.resolve())
