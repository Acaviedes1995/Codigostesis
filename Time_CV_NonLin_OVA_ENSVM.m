% CPU time of the Cross Validation OVA-Elastic Net SVM (nonlinear kernel)
% with the best parameters

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('ENSVM'))

load irisMn.mat
%load hayes_roth
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

FunPara.kerfPara.type = 'rbf';

FunPara.c1=2^2;
FunPara.c2=2^2;
FunPara.kerfPara.pars = 2^0;
for i=1:folds
    tst=perm(i:folds:m); % se fija la particion
    %Training data
    trn=setdiff(1:m,tst);
    Xa=X(trn,:);
    Ya=Y(trn,:);
    %Test data
    Xt=X(tst,:);
    Yot=Yo(tst,:);
    % Proceso OVA-ENSVM
    [Loss(i),bal_accu(i),Tf(i)]=Predi_OVA_ENSVM_dual(Xt,Yot,Xa,Ya,FunPara,T); %osqp
end

ACCU_dual=1-mean(Loss);
bACCU_dual=mean(bal_accu);
Cpu_dual=mean(Tf);

