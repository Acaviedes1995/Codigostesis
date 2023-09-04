% CPU time for Cross Validation OVA-NHSVM (nonlinear kernel) with the best
% parameters

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('NHSVM'))

%load irisMn.mat
%load hayes_roth
%load wineMn.mat
%load glassMn.mat
%load led7digit
%load vowelM
load pecesM
%load satimageMn
%load segmentMn
%load waveformM

T=max(Y);
folds=10;
[m,n]=size(X);

FunPara.kerfPara.type = 'rbf';
FunPara.c1=2^5;
FunPara.c2=2^(-4);
FunPara.kerfPara.pars = 2^(0);

for i=1:folds
    tst=perm(i:folds:m);
    %training data
    trn=setdiff(1:m,tst);
    Xa=X(trn,:);
    Ya=Y(trn,:);
    %Test data|
    Xt=X(tst,:);
    Yot=Yo(tst,:);
    [Loss(i,:),bal_accu(i,:),Tf(i)]=Predi_OVA_NHSVM_osqp(Xt,Yot,Xa,Ya,FunPara,T);
end
ACCU_vec=1-mean(Loss);
bACCU_vec=mean(bal_accu);
ACCU1=ACCU_vec(1);
ACCU2=ACCU_vec(2);
ACCU3=ACCU_vec(3);
bACCU1=bACCU_vec(1);
bACCU2=bACCU_vec(2);
bACCU3=bACCU_vec(3);
Cpu_time=mean(Tf);