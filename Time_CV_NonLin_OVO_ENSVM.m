% Cross Validation OVO-Elastic Net SVM (nonlinear kernel)

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('ENSVM'))

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

FunPara.c1=2^7;
FunPara.c2=2^7;
FunPara.kerfPara.pars = 2^4;
for i=1:folds
    tst=perm(i:folds:m); % se fija la particion
    %Training data
    trn=setdiff(1:m,tst);
    Xa=X(trn,:);
    Ya=Y(trn,:);
    %Test data
    Xt=X(tst,:);
    Yot=Yo(tst,:);
    % Proceso OVO-ENSVM
    [Loss(i),bal_accu(i),Tf(i)]=Predi_OVO_ENSVM_dual(Xt,Yot,Xa,Ya,FunPara,T); % osqp
end

ACCU_dual=1-mean(Loss);
bACCU_dual=mean(bal_accu);
Cpu_dual=mean(Tf);

