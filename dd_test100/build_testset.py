import json, random, re, sys, urllib.request, urllib.parse, os, csv
OUT = sys.argv[1]
API = "https://api.isic-archive.com/api/v2/images/search/"
# class -> (ISIC diagnosis_3 values, n images)
PLAN = {
 'akiec': (['Solar or actinic keratosis','Squamous cell carcinoma in situ'], 12),
 'bcc':   (['Basal cell carcinoma'], 15),
 'bkl':   (['Seborrheic keratosis','Solar lentigo','Lichen planus like keratosis'], 15),
 'df':    (['Dermatofibroma'], 10),
 'nv':    (['Nevus'], 18),
 'vasc':  (['Hemangioma'], 10),
 'mel':   (['Melanoma Invasive','Melanoma in situ'], 20),
}
def ham(i):  # HAM10000 (likely training data) ID range -> exclude to limit leakage
    n = int(re.search(r'(\d+)$', i).group(1)); return 24306 <= n <= 34320
def get(url):
    for _ in range(3):
        try: return json.load(urllib.request.urlopen(url, timeout=40))
        except Exception as e: err = e
    raise err
rows = []; rnd = random.Random(42)
for cls, (dxs, n) in PLAN.items():
    pool = []
    for dx in dxs:
        q = f'diagnosis_3:"{dx}" AND image_type:"dermoscopic"'
        url = API + "?" + urllib.parse.urlencode({'limit': 100, 'query': q})
        d = get(url)
        for r in d['results']:
            c = r['metadata']['clinical']
            if ham(r['isic_id']) or r['copyright_license'] != 'CC-0': continue
            pool.append((r['isic_id'], r['files']['full']['url'], dx, c.get('lesion_id') or r['isic_id'],
                         c.get('diagnosis_confirm_type','')))
    seen, uniq = set(), []
    rnd.shuffle(pool)
    for p in pool:
        if p[3] in seen: continue
        seen.add(p[3]); uniq.append(p)
    # keep a mix of sub-diagnoses
    chosen = uniq[:n]
    print(cls, 'pool', len(uniq), 'chosen', len(chosen))
    for iid, u, dx, lid, conf in chosen:
        path = f'{OUT}/images/{iid}.jpg'
        if not os.path.exists(path): urllib.request.urlretrieve(u, path)
        rows.append([f'images/{iid}.jpg', cls, iid, dx, conf, u])
with open(f'{OUT}/test_split.csv','w',newline='') as f:
    w = csv.writer(f); w.writerow(['image_path','label','isic_id','isic_diagnosis','confirm_type','source_url']); w.writerows(rows)
print('total', len(rows))
