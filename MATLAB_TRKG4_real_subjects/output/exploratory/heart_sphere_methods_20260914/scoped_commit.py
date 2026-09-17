from pathlib import Path
import subprocess,os,uuid,json,sys
root=Path(__file__).resolve().parents[4]
spec=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
paths=spec['paths']
map_path='Colab Notebooks/00.00_Карта_проекта.md'
section_title='### Эквивалентная сфера сердца: геометрия и электрический отклик'
def git(*args,env=None,data=None):
    result=subprocess.run(['git','-c','core.quotepath=false',*args],cwd=root,env=env,input=data,capture_output=True)
    if result.returncode:
        raise RuntimeError((result.stdout+result.stderr).decode('utf-8',errors='replace'))
    return result.stdout
index=Path(__file__).parent/('scoped_'+uuid.uuid4().hex+'.index')
env=os.environ.copy(); env['GIT_INDEX_FILE']=str(index)
try:
    base=git('rev-parse','HEAD').decode().strip()
    git('read-tree',base,env=env)
    git('add','-f','--',*paths,env=env) if spec.get('force_explicit_paths') else git('add','--',*paths,env=env)
    scope=set(paths)
    if spec.get('map_section'):
        original=git('show',base+':'+map_path).decode('utf-8')
        current=(root/map_path).read_text(encoding='utf-8')
        assert current.count(section_title)==1
        section=current[current.index(section_title):].strip()+'\n'
        if section_title in original:
            # Only replace the section owned by this task.
            desired=original[:original.index(section_title)]+section
        else:
            desired=original.rstrip()+'\n\n'+section
        blob=git('hash-object','-w','--stdin',data=desired.encode('utf-8')).decode().strip()
        git('update-index','--add','--cacheinfo','100644',blob,map_path,env=env)
        scope.add(map_path)
    for section in spec.get('owned_sections', []):
        target=section['path']; addition=section['text']
        current=(root/target).read_text(encoding='utf-8-sig')
        assert addition in current
        original=git('show',base+':'+target).decode('utf-8')
        assert addition.splitlines()[0] not in original
        desired=original.rstrip()+'\n\n'+addition
        blob=git('hash-object','-w','--stdin',data=desired.encode('utf-8')).decode().strip()
        git('update-index','--add','--cacheinfo','100644',blob,target,env=env)
        scope.add(target)
    git('diff','--cached','--check',env=env)
    changed=git('diff','--cached','--name-only',env=env).decode('utf-8').splitlines()
    assert set(changed)<=scope,changed
    if not changed:
        print(json.dumps({'status':'no_scoped_changes'},ensure_ascii=False))
        sys.exit(0)
    tree=git('write-tree',env=env).decode().strip()
    commit=git('commit-tree',tree,'-p',base,'-m',spec['message']).decode().strip()
    # Compare-and-swap prevents overwriting a concurrent task's commit.
    git('update-ref','-m',spec['message'],'HEAD',commit,base)
    # Synchronize only our entries; all other staged work is preserved.
    git('reset','--quiet',commit,'--',*changed)
    actual=git('show','--format=','--name-only',commit).decode('utf-8').splitlines()
    assert set(x for x in actual if x)==set(changed)
    print(json.dumps({'commit':commit,'parent':base,'files':changed},ensure_ascii=False,indent=2))
finally:
    if index.exists(): index.unlink()
