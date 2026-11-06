function test_bmstu_uniform_rho_fit
% Independent resistor-chain reference: measured middle resistance = 1.5*rho.
root=fileparts(fileparts(mfilename('fullpath')));addpath(fullfile(root,'src'));
for middle=[1.5,3]
    conductance=[2,1/middle,.5];incidence=[1,-1,0,0;0,1,-1,0;0,0,1,-1];
    K0=incidence'*diag(conductance)*incidence;
    G=3*eye(4);C=sparse([G,-G;-G,G]);
    target=10.95;expected=target/middle;
    a=trkg4_uniform_rho_fit(sparse(K0),C,target);
    assert(abs(a.rho_apparent_ohm_m-expected)<1e-8);
    assert(abs(a.predicted_Z_ohm-target)<1e-8);
end
rejected=false;
try,trkg4_uniform_rho_fit(sparse(K0),C,-1);catch,rejected=true;end
assert(rejected,'Nonpositive observation must be rejected');
fprintf('BMSTU_UNIFORM_RHO_TESTS_PASSED: two independent resistor geometries and invalid target\n');
end
