function tests=test_georg_two_source_readout
tests=functiontests(localfunctions);
end

function test_phase_hypotheses_and_gradients(testCase)
x=[.1,.2,.3];h=1e-5;
for scenario={'own_drive','in_phase','opposite_phase','quadrature_magnitude'}
 [value,J]=readout(x,scenario{1});fd=zeros(1,3);
 for j=1:3
  xp=x;xm=x;xp(j)=xp(j)+h;xm(j)=xm(j)-h;
  fd(j)=(readout(xp,scenario{1})-readout(xm,scenario{1}))/(2*h);
 end
 verifyLessThan(testCase,norm(fd-J)/norm(J),1e-8);
 verifyTrue(testCase,isfinite(value));
end
end

function test_source_sign_and_magnitude(testCase)
verifyEqual(testCase,trkg4_two_source_readout(70,-10,[],[],'own_drive'),70);
verifyEqual(testCase,trkg4_two_source_readout(70,-10,[],[],'in_phase'),60);
verifyEqual(testCase,trkg4_two_source_readout(70,-10,[],[],'opposite_phase'),80);
verifyEqual(testCase,trkg4_two_source_readout(70,-10,[],[],'quadrature_magnitude'),sqrt(5000),'AbsTol',1e-12);
end

function [z,J]=readout(x,scenario)
own=70*exp(x(1))+10*exp(x(2));counter=-8*exp(x(1))+3*exp(x(3));
Jown=[70*exp(x(1)),10*exp(x(2)),0];Jcounter=[-8*exp(x(1)),0,3*exp(x(3))];
[z,J]=trkg4_two_source_readout(own,counter,Jown,Jcounter,scenario);
end
