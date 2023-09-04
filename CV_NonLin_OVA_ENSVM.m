% Cross Validation OVA-ENSVM (nonlinear kernel)

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('ENSVM'))

%load irisMn.mat
%load hayes_roth.mat
%load wineMn.mat
%load glassMn.mat
%load led7digit
%load vowelM
%load pecesM
%load satimageMn
%load segmentMn
load waveformM

T=max(Y);
folds=10;
[m,n]=size(X);

FunPara.kerfPara.type = 'rbf';


for l=4:7
    l
    FunPara.c1=2^l;
    FunPara.c2=2^l;
    for j=6:7
        FunPara.kerfPara.pars = 2^j;
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
            [Loss(i),bal_accu(i)]=Predi_OVA_ENSVM_dual(Xt,Yot,Xa,Ya,FunPara,T);
        end
        ACCU(l+8,j+8)=1-mean(Loss);
        bACCU(l+8,j+8)=mean(bal_accu);
    end
   % descripcion='base satimageMn, kernel rbf';
   % save dataSatimageMn;
end
