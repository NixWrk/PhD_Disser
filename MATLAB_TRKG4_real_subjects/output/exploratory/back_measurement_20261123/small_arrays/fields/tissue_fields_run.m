function tissue_fields_run(manifestfile)
m=jsondecode(fileread(manifestfile));addpath(fullfile(m.root,'MATLAB_TRKG4_real_subjects','src'));
for j=1:numel(m.jobs)
 job=m.jobs(j);p=jsondecode(fileread(job.plan));
 assert(strcmp(trkg4_file_sha256(job.source_plan),job.source_plan_sha256));
 assert(strcmp(trkg4_file_sha256(p.shared_file),p.shared_sha256));
 assert(strcmp(trkg4_file_sha256(job.source_contacts),job.source_contacts_sha256));
 d=load(p.shared_file,'identity');
 assert(strcmp(d.identity.mesh_sha256,p.mesh_sha256));
 assert(strcmp(d.identity.plan_sha256,job.source_plan_sha256));clear d;
 trkg4_back_comparison(job.plan,'contacts');
 a=load(job.source_contacts,'C','plan_sha256');b=load(fullfile(p.work,'contacts.mat'),'C','plan_sha256');
 assert(strcmp(a.plan_sha256,job.source_plan_sha256));
 assert(strcmp(b.plan_sha256,trkg4_file_sha256(job.plan)));
 for k=1:numel(job.source_contact_indices)
  assert(nnz(a.C{job.source_contact_indices(k)}-b.C{k})==0,'Changed nominal contact operator');
 end
 clear a b;
 f=fopen(fullfile(p.work,'operator_identity.local.json'),'w');
 fprintf(f,'%s\n',jsonencode(struct('exact_contact_operator_identity',true,'count',numel(p.contacts))));fclose(f);
 trkg4_back_comparison(job.plan,'all');
end
end
