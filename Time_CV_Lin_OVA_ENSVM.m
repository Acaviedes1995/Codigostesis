% CPU time of the Cross Validation OVA-Elastic Net SVM (linear kernel)
% with the best parameters

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('ENSVM'))

%load irisMn.mat
%load hayes_roth
%load wineMn.mat
%load glassMn.mat
load led7digit
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
FunPara.c2=2^1;
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
    [Loss1(i),bal_accu1(i),T1(i)]=Predi_OVA_ENSVM(Xt,Yot,Xa,Ya,FunPara,T); % cvx
    [Loss2(i),bal_accu2(i),T2(i)]=Predi_OVA_ENSVM_dual(Xt,Yot,Xa,Ya,FunPara,T); %osqp
end

ACCU_primal=1-mean(Loss1);
bACCU_primal=mean(bal_accu1);
ACCU_dual=1-mean(Loss2);
bACCU_dual=mean(bal_accu2);
Cpu_primal=mean(T1);
Cpu_dual=mean(T2);

