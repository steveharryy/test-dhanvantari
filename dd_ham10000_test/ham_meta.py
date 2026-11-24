import json, re, urllib.request, urllib.parse, collections
url="https://api.isic-archive.com/api/v2/images/search/?"+urllib.parse.urlencode({'limit':100,'collections':212})
rows=[]
while url:
    for t in range(4):
        try: d=json.load(urllib.request.urlopen(url,timeout=60)); break
        except Exception as e: err=e
    else: raise err
    for r in d['results']:
        c=r['metadata']['clinical']
        rows.append({'id':r['isic_id'],'url':r['files']['full']['url'],'lesion':c.get('lesion_id'),
          'd1':c.get('diagnosis_1'),'d2':c.get('diagnosis_2'),'d3':c.get('diagnosis_3'),'d4':c.get('diagnosis_4'),
          'lic':r['copyright_license']})
    url=d.get('next')
json.dump(rows,open(f'{__import__("sys").argv[1]}/ham_meta.json','w'))
print(len(rows))
ham=[r for r in rows if 24306<=int(r['id'][-7:])<=34320]
print('in HAM id range',len(ham), 'lesions',len({r['lesion'] for r in ham}))
print(collections.Counter(r['d3'] for r in ham).most_common(30))
print('missing lesion',sum(1 for r in ham if not r['lesion']))
