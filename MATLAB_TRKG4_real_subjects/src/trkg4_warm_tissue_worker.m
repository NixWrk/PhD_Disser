function trkg4_warm_tissue_worker(planfile)
% Copy a frozen basis to a new study before enriching it; history stays intact.
p=jsondecode(fileread(planfile));old=jsondecode(fileread(p.warm_start_plan));
assert(strcmp(trkg4_file_sha256(p.warm_start_plan),p.warm_start_plan_sha256));
oldhash=p.warm_start_plan_sha256;plan_sha256=trkg4_file_sha256(planfile);
if ~isfile(fullfile(p.work,'basis.mat'))
 d=load(fullfile(old.work,'basis.mat'));assert(strcmp(d.plan_sha256,oldhash));
 basis=d.basis;reduced=d.reduced;count=d.count;clear d;
 save(fullfile(p.work,'basis.mat'),'basis','reduced','count','plan_sha256','-v7.3');clear basis reduced;
 d=load(fullfile(old.work,'contacts.mat'));assert(strcmp(d.plan_sha256,oldhash));C=d.C;clear d;
 save(fullfile(p.work,'contacts.mat'),'C','plan_sha256','-v7.3');clear C;
end
assert(strcmp(trkg4_file_sha256(which('trkg4_bmstu_inverse_worker')),p.underlying_worker_sha256));
trkg4_bmstu_inverse_worker(planfile);
end
