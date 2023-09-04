% Cross Validation OVA-NHSVM (linear kernel)

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
for l=-7:-7
    l
    FunPara.c1=2^l;
    for j=7:7
        FunPara.c2=2^(j);
        for i=1:folds
            tst=perm(i:folds:m);
            %training data
            trn=setdiff(1:m,tst);
            Xa=X(trn,:);
            Ya=Y(trn,:);
            %Test data|
            Xt=X(tst,:);
            Yot=Yo(tst,:);
            [Loss1(i),Loss2(i),Loss3(i),bal_accu1(i),bal_accu2(i),bal_accu3(i)]=Predi_OVANHSVM(Xt,Yot,Xa,Ya,FunPara,T);
        end
        ACCU1(l+8,j+8)=1-mean(Loss1);
        ACCU2(l+8,j+8)=1-mean(Loss2);
        ACCU3(l+8,j+8)=1-mean(Loss3);
        bACCU1(l+8,j+8)=mean(bal_accu1);
        bACCU2(l+8,j+8)=mean(bal_accu2);
        bACCU3(l+8,j+8)=mean(bal_accu3);
    end
end