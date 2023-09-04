% CPU time of Cross Validation OVO-NHSVM (linear kernel) with the best
% parameters

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('NHSVM'))

%load irisMn.mat
load hayes_roth
%load wineMn.mat
%load glassMn.mat
%load led7digit
%load vowelM
%load pecesM
%load satimageMn
%load segmentMn
%load waveformM


T=max(Y);
folds=10;
[m,n]=size(X);

FunPara.kerfPara.type = 'lin';

FunPara.c1=2^1;
FunPara.c2=2^2;
for k=1:folds
    tst=perm(k:folds:m);
    %training data
    trn=setdiff(1:m,tst);
    Xa=X(trn,:);
    Ya=Y(trn,:);
    %Test data
    Xt=X(tst,:);
    Yot=Yo(tst,:);
    %[Loss(k),bal_accu(k),Tf(k)]=Predi_OVO_NHSVM(Xt,Yot,Xa,Ya,FunPara,T);
    [Loss(k),bal_accu(k),Tf(k)]=Predi_OVO_NHSVM_osqp(Xt,Yot,Xa,Ya,FunPara,T);
end
ACCU=1-mean(Loss);
bACCU=mean(bal_accu);
Cpu_time=mean(Tf);