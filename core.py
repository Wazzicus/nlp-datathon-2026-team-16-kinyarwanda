"""Shared small-data baselines; no paid APIs or automatic translation."""
import json, re, hashlib
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def labels():
    return ['positive','negative','neutral']
def fingerprint(df):
    return hashlib.sha256(df.to_csv(index=False).encode('utf-8')).hexdigest()
def slot_memory(train):
    counts=Counter()
    for text, raw in zip(train.text,train.slots_json):
        for s in json.loads(raw): counts[(s['value'].casefold(),s['type'])]+=1
    # One deterministic type per known surface form; use training data only.
    values={}
    for (value, kind), count in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        values.setdefault(value,kind)
    return values
def extract(text,memory):
    candidates=[]
    for value, kind in memory.items():
        for match in re.finditer(r'(?<!\w)'+re.escape(value)+r'(?!\w)',text,re.IGNORECASE):
            candidates.append({'type':kind,'start':match.start(),'end':match.end(),'value':match.group()})
    accepted=[]
    for s in sorted(candidates,key=lambda s:(-(s['end']-s['start']),s['start'],s['type'])):
        if not any(s['start']<a['end'] and a['start']<s['end'] for a in accepted):accepted.append(s)
    return sorted(accepted,key=lambda s:s['start'])
def slot_keys(slots):
    return {(s['type'],s['start'],s['end']) for s in slots}
def classification_score(model,frame,ordered):
    pred=model.predict(frame.text)
    return {'accuracy':accuracy_score(frame.label,pred),'macro_f1':f1_score(frame.label,pred,labels=ordered,average='macro',zero_division=0),
    'report':classification_report(frame.label,pred,labels=ordered,output_dict=True,zero_division=0),
    'confusion_matrix':confusion_matrix(frame.label,pred,labels=ordered).tolist(),'label_order':ordered}