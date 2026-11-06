function answer=trkg4_uniform_rho_fit(K0,C,target)
% One positive homogeneous resistivity; fixed finite-contact matrix.
% Lateral port order A,M,N,B; the physical core is unchanged.
P=[1,0;0,1;0,-1;-1,0];
[T,~,unitqc]=trkg4_lateral_cem_response(K0,C,{K0},1,P,[],1);
unit=T(2,1);assert(unit>0&&isfinite(target)&&target>0);
initial=target/unit;rho=initial;history=struct([]);converged=false;
tolerance=1e-5+1e-7*abs(target);
for i=1:6
    [T,J,qc]=trkg4_lateral_cem_response(K0/rho,C,{K0},rho,P,1,1);
    z=T(2,1);slope=J(2,1,1);residual=z-target;
    assert(isfinite(slope)&&slope>0);
    step=struct('rho_ohm_m',rho,'Z_ohm',z,'residual_ohm',residual,...
        'dZ_dlogrho_ohm',slope,'system_relative_residual',qc.system_relative_residual,...
        'reciprocity_ohm',qc.reciprocity_ohm);
    if isempty(history),history=step;else,history(end+1)=step;end
    fprintf(' rho=%.9g residual=%.6g Ohm\n',rho,residual);
    if abs(residual)<=tolerance,converged=true;break;end
    rho=rho*exp(max(-1,min(1,-residual/slope)));
end
assert(converged,'Uniform-rho fit did not meet the fixed impedance tolerance');
answer=struct('observed_Z_ohm',target,'rho_apparent_ohm_m',rho,...
    'Z_unit_rho_ohm',unit,'linear_ratio_rho_ohm_m',initial,...
    'fixed_contact_correction_relative',(rho-initial)/initial,...
    'predicted_Z_ohm',z,'residual_ohm',residual,'tolerance_ohm',tolerance,...
    'dZ_dlogrho_ohm',slope,'iterations',history,'unit_response_qc',unitqc);
end
