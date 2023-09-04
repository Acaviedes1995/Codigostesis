% Cross Validation OVO-NHSVM (linear kernel)

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
for l=-7:7
    l
    FunPara.c1=2^l;
    for j=-7:7
        FunPara.c2=2^j;
        for k=1:folds
            tst=perm(k:folds:m);
            %training data
            trn=setdiff(1:m,tst);
            Xa=X(trn,:);
            Ya=Y(trn,:);
            %Test data
            Xt=X(tst,:);
            Yot=Yo(tst,:);
            [Loss(k),bal_accu(k)]=Predi_OVO_NHSVM(Xt,Yot,Xa,Ya,FunPara,T);
        end
        ACCU(l+8,j+8)=1-mean(Loss);
        bACCU(l+8,j+8)=mean(bal_accu);
    end
end