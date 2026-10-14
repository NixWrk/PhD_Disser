function [value,J]=trkg4_two_source_readout(own,counter,Jown,Jcounter,scenario)
% Declared equal-current source/readout hypotheses, not fitted channel gains.
assert(isfinite(own)&&isfinite(counter)&&isequal(size(Jown),size(Jcounter)));
switch scenario
 case 'own_drive',value=own;J=Jown;
 case 'in_phase',value=own+counter;J=Jown+Jcounter;
 case 'opposite_phase',value=own-counter;J=Jown-Jcounter;
 case 'quadrature_magnitude'
  value=hypot(own,counter);assert(value>0,'Magnitude derivative is undefined at zero');
  J=(own*Jown+counter*Jcounter)/value;
 otherwise,error('georg:sourceScenario','Unknown source/readout scenario');
end
end
