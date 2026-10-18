"""Bounded, allowlisted operational metadata; never queries or source bodies."""
import json,logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ALLOWED={'pid','generation','operation_id','project_id','state','error','backend','elapsed_ms','count'}
EVENTS={'startup','operation','shutdown','model_failure'}

def metadata_logger(runtime:Path):
    directory=runtime/'logs';directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    logger=logging.getLogger('cmr.metadata.'+str(runtime));logger.propagate=False;logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler=RotatingFileHandler(directory/'metadata.log',maxBytes=10*1024*1024,backupCount=3,encoding='utf-8',delay=True)
        handler.setFormatter(logging.Formatter('%(message)s'));logger.addHandler(handler)
    return logger

def emit(logger,event,**fields):
    if event not in EVENTS or set(fields)-ALLOWED:
        raise ValueError('unapproved metadata')
    if any(not isinstance(v,(str,int,float,bool,type(None))) or isinstance(v,str) and len(v)>128 for v in fields.values()):
        raise ValueError('unapproved metadata value')
    logger.info(json.dumps({'event':event,**fields},ensure_ascii=False,allow_nan=False))
