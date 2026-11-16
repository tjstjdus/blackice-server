"""Shared training/inference definitions. Raw weather retrieval is external."""
from pathlib import Path
import sys,json
R=Path(__file__).resolve().parent
if (R/'deps').exists():sys.path.insert(0,str(R/'deps'))
import numpy as np,pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,brier_score_loss,confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

class FeatureBuilder(TransformerMixin,BaseEstimator):
 def __init__(self,columns):self.columns=columns
 def fit(self,X,y=None):self.transform(X);return self
 def transform(self,X):
  missing=set(self.columns)-set(X.columns)
  if missing:raise ValueError('Missing required columns: '+str(sorted(missing)))
  z=X[list(self.columns)].copy()
  if 'x_wind_direction_deg' in z:
   a=np.deg2rad(z.pop('x_wind_direction_deg').astype(float))
   z['x_wind_sin']=np.sin(a);z['x_wind_cos']=np.cos(a)
   calm=z.x_wind_speed_ms.eq(0)
   z.loc[calm,['x_wind_sin','x_wind_cos']]=0
  return z.to_numpy(dtype=np.float32)

def candidates(plan,alg):
 key={'XGBoost':'xgboost','RandomForest':'random_forest','LogisticRegression':'logistic'}[alg]
 return [(i,w,{**plan[key+'_fixed'],**p}) for i,p in enumerate(plan[key+'_candidates']) for w in ['none','balanced']]

def fit_model(alg,params,weight,cols,X,y):
 import warnings
 from sklearn.exceptions import ConvergenceWarning
 if len(np.unique(y))!=2:raise ValueError('Both training classes required')
 raw=FeatureBuilder(cols).transform(X)
 if np.isnan(raw).all(axis=0).any():raise ValueError('All-missing training feature; user decision required')
 if np.isinf(raw).any():raise ValueError('Infinite input')
 del raw
 estimator={'XGBoost':XGBClassifier,'RandomForest':RandomForestClassifier,'LogisticRegression':LogisticRegression}[alg](**params,**({'n_jobs':4} if alg!='LogisticRegression' else {}))
 steps=[('features',FeatureBuilder(cols)),('imputer',SimpleImputer(strategy='median'))]
 if alg=='LogisticRegression':steps.append(('scale',StandardScaler()))
 steps.append(('model',estimator));model=Pipeline(steps)
 sw=compute_sample_weight('balanced',y) if weight=='balanced' else np.ones(len(y))
 with warnings.catch_warnings():
  warnings.simplefilter('error',ConvergenceWarning)
  model.fit(X,y,model__sample_weight=sw)
 return model

def metrics(y,p,t=.5):
 pred=np.asarray(p)>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'n':len(y),'positive':int(np.sum(y)),'threshold':float(t),'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'AP':float(average_precision_score(y,p)) if np.sum(y)>0 else None,'ROC_AUC':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,'Brier':float(brier_score_loss(y,p)),'TP':int(tp),'FN':int(fn),'FP':int(fp),'TN':int(tn)}

def threshold_from_validation(y,p):
 from sklearn.metrics import precision_recall_curve
 precision,recall,ts=precision_recall_curve(y,p)
 f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-30)
 idx=np.lexsort((ts,recall[:-1],f))[-1]
 return float(ts[idx])

def feature_sets():
 m=json.loads((R/'source/feature_manifest.json').read_text(encoding='utf-8'))['feature_sets']
 weather=[x for x in m['terrain3_weather'] if x not in m['terrain3']]
 return {**{k:m[k] for k in ['terrain3','terrain7','terrain12']},'weather':weather,**{k:m[k] for k in ['terrain3_weather','terrain7_weather','terrain12_weather']}}

def save_json(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def predict_rows(bundle_path,rows):
 """Offline prediction using already retrieved, raw named features."""
 import joblib
 bundle_path=Path(bundle_path)
 b=joblib.load(bundle_path);d=pd.DataFrame(rows)
 cols=b['columns'];rawcols=[c for c in cols if c!='x_stage1_score'];absent=set(rawcols)-set(d.columns)
 if absent:raise ValueError('Missing required fields: '+str(sorted(absent)))
 valid=~d[rawcols].isna().all(axis=1)
 if 'stage1_file' in b:
  first=joblib.load(bundle_path.parent/b['stage1_file'])
  d['x_stage1_score']=np.nan
  if valid.any():d.loc[valid,'x_stage1_score']=first['pipeline'].predict_proba(d.loc[valid])[:,1]
 elif 'stage1_model_path' in b:
  first=joblib.load(R/b['stage1_model_path']/'model.joblib')
  d['x_stage1_score']=np.nan
  if valid.any():d.loc[valid,'x_stage1_score']=first['pipeline'].predict_proba(d.loc[valid])[:,1]
 scores=np.full(len(d),np.nan)
 if valid.any():scores[valid]=b['pipeline'].predict_proba(d.loc[valid])[:,1]
 return [{'score':None if not np.isfinite(s) else float(s),'classification':None if not np.isfinite(s) else int(s>=b['selected_threshold']),'status':'insufficient_input' if not np.isfinite(s) else 'ok','calibrated':False,'risk_grade':None,'model_version':b['model_version'],'target_meaning':b['target_meaning']} for s in scores]
