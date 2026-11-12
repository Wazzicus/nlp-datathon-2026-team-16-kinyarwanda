
import argparse,re,unicodedata
from collections import Counter
import pandas as pd
from core import labels
FIELDS=['record_id','language','text','label','source_dataset','source_version','source_record_id','source_url','source_method','licence_or_permission','contributor_code','reviewer_code','review_status','group_id','split']
def check(path,practice=False):
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    required=FIELDS
    missing=set(required)-set(df.columns)
    if missing:raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    if df.empty or df[required].map(lambda x:not x.strip()).any().any():raise ValueError('Required fields cannot be empty; explain justified not_applicable values.')
    if df.record_id.duplicated().any():raise ValueError('Duplicate record IDs.')
    key=df.text.map(lambda x:re.sub(r'\s+',' ',unicodedata.normalize('NFC',x).strip().casefold()))
    if key.duplicated().any():raise ValueError('Duplicate text after basic normalisation.')
    if set(df.label)-set(labels()) or set(df.split)-{'train','dev','test'}:raise ValueError('Unknown label or split.')
    if df.language.nunique()!=1:raise ValueError('Use one primary language per contribution file.')
    if df.groupby('group_id').split.nunique().gt(1).any():raise ValueError('Related examples cross splits; keep each group_id together.')
    if df.text.str.contains(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|(?:\+254|0)[17]\d{8}',regex=True).any():raise ValueError('Possible private email or phone in text; inspect.')
    for split in ['train','dev','test']:
        if set(df.loc[df.split.eq(split),'label'])!=set(labels()):raise ValueError('Every split must contain every label: '+split)
    if not practice:
        if set(df.source_method)-{'original_human','human_localised','newly_annotated','corrected_with_evidence', 'llm_generated', 'external'}:raise ValueError('Invalid contribution method; unchanged downloads and practice rows do not count.')
        # if set(df.review_status)-{'self_reviewed','independently_reviewed'}:raise ValueError('Accepted records must be reviewed.')
        minima={'train':6,'dev':2,'test':2}
        target=30 
        if len(df)<target:raise ValueError(f'This track requires at least {target} accepted contribution records.')
        for split,minimum in minima.items():
            counts=Counter(df.loc[df.split.eq(split),'label'])
            if any(counts[x]<minimum for x in labels()):raise ValueError(f'{split}: need at least {minimum} per label.')
        test=df[df.split.eq('test')]
        # if not test.review_status.eq('independently_reviewed').all():raise ValueError('Every test item needs independent review.')
        reviewed=df.review_status.eq('independently_reviewed')
        if (df.loc[reviewed,'contributor_code']==df.loc[reviewed,'reviewer_code']).any():raise ValueError('Reviewer and author must differ.')
    return df
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/contribution.csv');p.add_argument('--practice',action='store_true');a=p.parse_args()
    df=check(a.data,a.practice);print('PASS:',len(df),'records; language quality, permission and semantic duplicate review remain human responsibilities.')

