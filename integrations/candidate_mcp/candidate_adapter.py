"""Candidate-only projection. No business file, index, model or network client imports."""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

WARNINGS=['CANDIDATE_PILOT','SUPPORT_NOT_CHECKED','完整支持未检查']
FLAGS={'retrieval_role':'related_candidate','support_status':'not_checked'}
HASH=re.compile('[0-9a-f]{64}')
SAFE_ERRORS={'ENVIRONMENT_CHANGED','INPUT_LOCK_CHANGED','STARTUP_FLAGS_CHANGED','DAEMON_UNVERIFIED',
 'SOURCE_CHANGED','STALE_SOURCE','INDEX_NOT_READY','MODEL_UNAVAILABLE','SCOPE_DENIED','INVALID_REQUEST',
 'CANDIDATE_CONTRACT_MISMATCH','RETRIEVAL_RESPONSE_INVALID','GROUP_VERIFICATION_EXPIRED',
 'SOURCE_NOT_CONFIGURED','PROCESS_IDENTITY_UNVERIFIED','CLI_TIMEOUT','CLI_OUTPUT_LIMIT',
 'CLI_FAILED','CLI_RESPONSE_INVALID','CLI_CANCELLED','DEADLINE_EXCEEDED'}


class AdapterError(RuntimeError):
    pass


def require(condition,code='RETRIEVAL_RESPONSE_INVALID'):
    if not condition:raise AdapterError(code)


def relative_path(value):
    require(isinstance(value,str) and 0<len(value)<=1024 and '\0' not in value,'SCOPE_DENIED')
    p=PurePosixPath(value)
    require(not p.is_absolute() and str(p)==value and value!='.' and not any(
        x in {'..','.git','.env','99-inbox','90-archive'} or x.startswith('.env.') for x in p.parts)
        and not any(x in value for x in ['\\','*','?','codex-memory-protocol','codex-memory-index']), 'SCOPE_DENIED')
    return value


class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,frozen=True)


class Grant(Strict):
    source_id:Literal['ca-memory','ca-docs','ca-code']
    relative_path:str
    file_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    chunk_id:str=Field(pattern=r'^[0-9a-f]{64}$')
    line_start:int=Field(ge=1)
    line_end:int=Field(ge=1)
    @field_validator('relative_path')
    @classmethod
    def path_valid(cls,value):return relative_path(value)


class Policy(Strict):
    schema_version:Literal['candidate-cloud-policy-v2']=Field(alias='schema')
    approved:bool=False
    adapter_seal_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    r4_input_hash:str=Field(pattern=r'^[0-9a-f]{64}$')
    project_id:Literal['composing-agent']
    max_context_chars:int=Field(ge=1,le=12000)
    files:list[Grant]=Field(max_length=256)


@dataclass(frozen=True)
class PolicySnapshot:
    policy:Policy
    state:tuple


class PolicyFile:
    def __init__(self,path,adapter_seal,r4_input_hash):
        self.path=Path(path);self.adapter_seal=adapter_seal;self.r4_input_hash=r4_input_hash

    def read(self):
        try:
            p=self.path
            require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in [p,*p.parents]),'CLOUD_POLICY_INVALID')
            require(stat.S_ISREG(p.lstat().st_mode),'CLOUD_POLICY_INVALID')
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as stream:
                before=os.fstat(stream.fileno())
                require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid()
                        and stat.S_IMODE(before.st_mode)==0o600 and before.st_size<=131072,'CLOUD_POLICY_INVALID')
                payload=stream.read(131073);after=os.fstat(stream.fileno())
            fields=('st_dev','st_ino','st_mode','st_uid','st_size','st_mtime_ns','st_ctime_ns')
            require(all(getattr(before,k)==getattr(after,k) for k in fields) and len(payload)<=131072,'CLOUD_POLICY_INVALID')
            state=(before.st_dev,before.st_ino,before.st_mtime_ns,before.st_ctime_ns,hashlib.sha256(payload).hexdigest())
            return PolicySnapshot(Policy.model_validate_json(payload),state)
        except AdapterError:raise
        except Exception:raise AdapterError('CLOUD_POLICY_INVALID') from None

    def snapshot(self):
        snap=self.read();p=snap.policy
        require(p.approved,'CLOUD_SCOPE_NOT_APPROVED')
        require(p.adapter_seal_sha256==self.adapter_seal and p.r4_input_hash==self.r4_input_hash,'CLOUD_POLICY_INVALID')
        require(all(g.line_start<=g.line_end for g in p.files),'CLOUD_POLICY_INVALID')
        require(len({g.chunk_id for g in p.files})==len(p.files),'CLOUD_POLICY_INVALID')
        return snap

    def check(self,snapshot):
        try:require(self.snapshot()==snapshot,'CLOUD_POLICY_CHANGED')
        except AdapterError:raise AdapterError('CLOUD_POLICY_CHANGED') from None


class Empty(Strict):pass
class MemoryQuery(Strict):
    query:str=Field(min_length=1,max_length=32768)
class CodeQuery(MemoryQuery):
    literal:str|None=Field(default=None,max_length=256)
class ContextQuery(Strict):
    chunk_id:str=Field(pattern=r'^[0-9a-f]{64}$')
    max_chars:int=Field(default=4000,ge=1,le=12000)


def reject_error(value):
    if isinstance(value,dict) and value.get('error') is not None:
        code=value['error']
        raise AdapterError(code if isinstance(code,str) and code in SAFE_ERRORS else 'RETRIEVAL_RESPONSE_INVALID')


class Adapter:
    def __init__(self,policy,cli,sources,clock=time.monotonic):
        self.policy,self.cli,self.sources,self.clock=policy,cli,sources,clock
        self.candidates=OrderedDict();self.snapshot_key=None;self.policy_snapshot=None;self.lock=asyncio.Lock()

    def envelope(self,data):
        reject_error(data)
        require(isinstance(data,dict) and isinstance(data.get('warnings'),list)
                and all(x in data['warnings'] for x in WARNINGS),'CANDIDATE_CONTRACT_MISMATCH')
        require(isinstance(data.get('generation'),str) and 0<len(data['generation'])<=256
                and type(data.get('table_version')) is int and data['table_version']>=0)
        require(type(data.get('truncated')) is bool)
        return {'generation':data['generation'],'table_version':data['table_version'],
                'warnings':list(WARNINGS),'truncated':data['truncated']}

    def item(self,value,kind=None):
        reject_error(value)
        require(isinstance(value,dict))
        require(value.get('project_id')=='composing-agent','SCOPE_DENIED')
        source=value.get('source_id');require(source in self.sources,'SCOPE_DENIED')
        if kind=='code':require(source=='ca-code','SCOPE_DENIED')
        elif kind=='memory':require(source in {'ca-memory','ca-docs'},'SCOPE_DENIED')
        require(value.get('worktree_id')==self.sources[source]['worktree_id'],'SCOPE_DENIED')
        path=relative_path(value.get('relative_path'))
        require(value.get('path')==str(Path(self.sources[source]['root'])/path),'SCOPE_DENIED')
        require(all(isinstance(value.get(k),str) and HASH.fullmatch(value[k]) for k in ['chunk_id','file_sha256']))
        require(type(value.get('line_start')) is int and type(value.get('line_end')) is int
                and 1<=value['line_start']<=value['line_end'])
        require(all(value.get(k)==v for k,v in FLAGS.items()),'CANDIDATE_CONTRACT_MISMATCH')
        state=value.get('retrieval_state');basis=value.get('evidence_basis')
        require(state in {'eligible','source_current'} and type(value.get('current_evidence')) is bool)
        require((state=='source_current' and value['current_evidence'] and basis=='verified_source_bytes') or
                (state=='eligible' and not value['current_evidence'] and basis=='dated_memory_assertion'))
        status=value.get('status');require(status in {None,'active','draft'})
        indexed=value.get('indexed_at');require(indexed is None or isinstance(indexed,str) and len(indexed)<=64)
        result={k:value[k] for k in ['chunk_id','project_id','source_id','worktree_id','relative_path','line_start','line_end','file_sha256',
                                     'retrieval_state','current_evidence','evidence_basis']}
        result.update(FLAGS);result.update(status=status,indexed_at=indexed)
        for key in ['rrf_score','rerank_score']:
            if key in value:
                score=value[key];require(type(score) in {int,float} and math.isfinite(score) and abs(score)<=1e6)
                result[key]=score
        return result

    @staticmethod
    def granted(item,policy):
        return any(g.source_id==item['source_id'] and g.relative_path==item['relative_path']
                   and g.file_sha256==item['file_sha256'] and g.chunk_id==item['chunk_id']
                   and g.line_start==item['line_start'] and g.line_end==item['line_end'] for g in policy.files)

    def project_search(self,data,kind,policy):
        out=self.envelope(data)
        require(isinstance(data.get('results'),list) and len(data['results'])<=8)
        require(data.get('mode') in {'hybrid','hybrid_reranked','hybrid_without_reranker','literal','lexical','vector'})
        out['mode']=data['mode'];out['results']=[]
        seen=set()
        for value in data['results']:
            entry=self.item(value,kind)
            require(entry['chunk_id'] not in seen);seen.add(entry['chunk_id'])
            if self.granted(entry,policy):out['results'].append(entry)
            else:out['truncated']=True
        return out

    def project_context(self,data,issued,limit):
        out=self.envelope(data)
        require(isinstance(data.get('items'),list) and len(data['items'])==1)
        entry=self.item(data['items'][0]);saved=issued['item']
        identity=['chunk_id','project_id','source_id','worktree_id','relative_path','line_start','line_end','file_sha256',
                  'retrieval_state','current_evidence','evidence_basis','status','indexed_at']
        require(all(entry[k]==saved[k] for k in identity) and
                (out['generation'],out['table_version'])==issued['snapshot'],'SNAPSHOT_OR_CANDIDATE_CHANGED')
        text=data['items'][0].get('text');require(isinstance(text,str) and len(text)<=limit)
        partial=out['truncated'] or not text
        require(data.get('readback_status')==('partial' if partial else 'complete'))
        out['readback_status']='partial' if partial else 'complete';out['truncated']=partial
        out['items']=[entry|{'text':text}]
        return out

    def project_status(self,data):
        require(isinstance(data,dict) and all(w in data.get('warnings',[]) for w in WARNINGS),'CANDIDATE_CONTRACT_MISMATCH')
        require(type(data.get('pid')) is int and data['pid']>1 and data.get('dimension')==768)
        require(all(type(data.get(k)) is bool for k in ['loading','syncing']))
        require(data.get('error') is None or data['error'] in SAFE_ERRORS)
        rank=data.get('reranker');require(isinstance(rank,dict))
        require(type(rank.get('enabled')) is bool and type(rank.get('ready')) is bool)
        require(rank.get('error') is None or rank['error'] in SAFE_ERRORS)
        require(data.get('backend') in {'cpu','mps'} and rank.get('backend') in {'cpu','mps'})
        for x in [data.get('generation'),data.get('embedding_profile'),rank.get('profile')]:
            require(x is None or isinstance(x,str) and len(x)<=256)
        sources=data.get('sources');require(isinstance(sources,dict) and set(sources)<=set(self.sources))
        states={}
        for name,value in sources.items():
            require(isinstance(value,dict) and value.get('state') in {'ready','pending','unavailable','not_indexed','not_ready','syncing','error'})
            states[name]=value['state']
        return {'project_id':'composing-agent','warnings':list(WARNINGS),**FLAGS,
                'pid':data['pid'],'dimension':768,'backend':data['backend'],'loading':data['loading'],
                'syncing':data['syncing'],'error':data['error'],'generation':data.get('generation'),
                'embedding_profile':data.get('embedding_profile'),
                'reranker':{k:rank.get(k) for k in ['enabled','ready','error','profile','backend']},'sources':states}

    async def dispatch(self,name,arguments):
        models={'candidate_status':Empty,'candidate_search_memory':MemoryQuery,
                'candidate_search_code':CodeQuery,'candidate_get_context':ContextQuery}
        require(name in models,'SCOPE_DENIED')
        try:args=models[name].model_validate(arguments)
        except ValidationError:raise AdapterError('INVALID_REQUEST') from None
        require(not self.lock.locked(),'ADAPTER_BUSY')
        async with self.lock:
            try:
                snap=self.policy.snapshot();end=self.clock()+35
                # Candidate authority belongs to the exact issuing policy,
                # including its file identity, not only the index generation.
                if snap!=self.policy_snapshot:
                    self.candidates.clear();self.snapshot_key=None
                    self.policy_snapshot=snap
                pending=None
                if name=='candidate_status':
                    out=self.project_status(await asyncio.wait_for(self.cli(['status','--project','composing-agent'],end),30))
                elif name.startswith('candidate_search_'):
                    kind='code' if name.endswith('code') else 'memory'
                    command=['search','--project','composing-agent','--kind',kind,'--query',args.query]
                    if kind=='code':command+=['--source','ca-code']
                    if kind=='code' and args.literal is not None:command+=['--literal',args.literal]
                    raw=await asyncio.wait_for(self.cli(command,end),30)
                    out=self.project_search(raw,kind,snap.policy);pending=out
                else:
                    issued=self.candidates.get(args.chunk_id)
                    require(issued is not None and issued['policy']==snap and issued['expires']>self.clock(),'CANDIDATE_NOT_ISSUED')
                    require(self.granted(issued['item'],snap.policy),'CLOUD_SCOPE_CHANGED')
                    limit=min(args.max_chars,snap.policy.max_context_chars)
                    raw=await asyncio.wait_for(self.cli(['context','--project','composing-agent','--chunk-id',args.chunk_id,'--max-chars',str(limit)],end),30)
                    out=self.project_context(raw,issued,limit)
                require(self.clock()<end,'DEADLINE_EXCEEDED');self.policy.check(snap)
                require(len(json.dumps(out,ensure_ascii=False,allow_nan=False).encode())<=131072,'OUTPUT_LIMIT')
                if pending is not None:
                    key=(out['generation'],out['table_version'])
                    if key!=self.snapshot_key:self.candidates.clear()
                    self.snapshot_key=key
                    for entry in out['results']:
                        self.candidates.pop(entry['chunk_id'],None)
                        self.candidates[entry['chunk_id']]={'item':entry,'snapshot':key,'policy':snap,'expires':self.clock()+300}
                    while len(self.candidates)>64:self.candidates.popitem(last=False)
                return out
            except asyncio.CancelledError:
                self.candidates.clear();raise
            except asyncio.TimeoutError:
                self.candidates.clear();raise AdapterError('CLI_TIMEOUT') from None
            except AdapterError:
                self.candidates.clear();raise
            except Exception:
                self.candidates.clear();raise AdapterError('ADAPTER_INTERNAL_ERROR') from None
