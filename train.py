import argparse,json
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from validate import check
from core import labels,fingerprint,classification_score
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/contribution.csv');p.add_argument('--practice',action='store_true');a=p.parse_args()
    df=check(a.data,a.practice)
    tr=df[df.split.eq('train')]
    dev=df[df.split.eq('dev')]

    print("Training records:", len(tr))
    print("Development records:", len(dev))

    print("\nTraining distribution:")
    print(tr.label.value_counts())

    print("\nDevelopment distribution:")
    print(dev.label.value_counts())

    model = make_pipeline(
    TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5)
    ),
    LinearSVC(C=1.0, random_state=42)
    )
  
    model.fit(tr.text,tr.label)
    dummy=DummyClassifier(strategy='most_frequent').fit(tr.text,tr.label)
    
    pred = model.predict(dev.text)
    
    errors = dev[pred != dev.label].copy()
    errors["predicted"] = pred[pred != dev.label]
    
    print(errors[["text", "label", "predicted"]].to_string(index=False))
        
    memory= {}
    pack={'model':model,'dummy':dummy,'language':df.language.iloc[0],'practice':a.practice,'track':'sentiment','data_sha256':fingerprint(df),'slot_memory':memory}
    out=Path('artifacts/baseline');out.mkdir(exist_ok=True,parents=True);joblib.dump(pack,out/'model.joblib')
    result={'track':'sentiment','practice':a.practice,'model':classification_score(model,dev,labels()),'majority_baseline':classification_score(dummy,dev,labels())}
    print("\nClassification Report:")
    print(result['model']['report'])
    (out/'dev_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Saved {out}/model.joblib and dev_metrics.json. Dev macro-F1: {result['model']['macro_f1']:.3f}. Freeze choices before test evaluation.")

