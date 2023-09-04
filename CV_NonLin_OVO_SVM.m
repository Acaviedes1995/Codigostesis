% Cross Validation OVO-SVM (nonlinear kernel)

clear all
addpath(genpath('data_set_multi')) 
addpath(genpath('SVM'))

%load irisMn.mat
%load hayes_roth
%load wineMn.mat
%load glassMn.mat
%load vowelM
%load pecesM
%load led7digit
load satimageMn
%load segmentMn
%load waveformM

T=max(Y);
folds=10;
[m,n]=size(X);

FunPara.kerfPara.type = 'rbf';

for l=5:7
    l
    FunPara.c=2^l;
    for j=-3:7
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
            % Proceso OVO-SVM
            [Loss(i),bal_accu(i)]=Predi_OVO_SVM(Xt,Yot,Xa,Ya,FunPara,T);
        end
        ACCU(l+8,j+8)=1-mean(Loss);
        bACCU(l+8,j+8)=mean(bal_accu);
    end
end
