import json, random, csv, os, sys, urllib.request, collections
from concurrent.futures import ThreadPoolExecutor
S, OUT = sys.argv[1], sys.argv[2]
rows=[r for r in json.load(open(f'{S}/ham_meta.json')) if 24306<=int(r['id'][-7:])<=34320]
def cls(r):
    d3=r['d3']
    return {'Nevus':'nv','Melanoma, NOS':'mel','Pigmented benign keratosis':'bkl','Basal cell carcinoma':'bcc',
            'Squamous cell carcinoma, NOS':'akiec','Solar or actinic keratosis':'akiec','Dermatofibroma':'df',None:'vasc'}[d3]
for r in rows: r['cls']=cls(r)
assert all(r['d3'] or 'vascular' in (r['d2'] or '').lower() for r in rows), 'unexpected None-d3 not vascular'
# lesion-level split: each lesion goes entirely to test or train; ~15% of lesions per class, seed 42
les=collections.defaultdict(list)
for r in rows: les[r['lesion']].append(r)
bycls=collections.defaultdict(list)
for l,rs in les.items(): bycls[collections.Counter(x['cls'] for x in rs).most_common(1)[0][0]].append(l)
rnd=random.Random(42); test=set()
for c,ls in sorted(bycls.items()):
    ls.sort(); rnd.shuffle(ls); test.update(ls[:round(0.15*len(ls))])
tst=[r for r in rows if r['lesion'] in test]
print('lesions test',len(test),'images test',len(tst),dict(collections.Counter(r['cls'] for r in tst)))
def dl(r):
    p=f'{OUT}/images/{r["id"]}.jpg'
    if not os.path.exists(p):
        for _ in range(4):
            try: urllib.request.urlretrieve(r['url'],p); return
            except Exception: pass
        raise RuntimeError(r['id'])
with ThreadPoolExecutor(12) as ex: list(ex.map(dl,tst))
with open(f'{OUT}/test_split.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['image_path','label','isic_id','lesion_id','isic_diagnosis'])
    for r in sorted(tst,key=lambda r:r['id']): w.writerow([f'images/{r["id"]}.jpg',r['cls'],r['id'],r['lesion'],r['d3'] or r['d2']])
