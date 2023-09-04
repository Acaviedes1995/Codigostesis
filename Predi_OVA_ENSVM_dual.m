function [Loss,Bal_accu,Tfinal]=Predi_OVA_ENSVM_dual(Xt,Yot,Xa,Ya,FunPara,T)

m=size(Xa,1);
mt=size(Xt,1);
Prediction=-ones(mt,T);
tf=zeros(T,1);

for k=1:T
    fin1=find(Ya==k); 
    fin2=setdiff([1:length(Ya)]',[fin1]);
    mk=length(fin1);
    mkk=m-mk;
    A=Xa(fin1,:); % class k
    B=Xa(fin2,:); % Other classes
    Xtr=[A;B];
    Ytr=[ones(mk,1);-ones(mkk,1)];
    %[Predic_y,tf(k),Val_testX]=ENSVM_dual_cvx(Xtr,Ytr,Xt,FunPara);
    [Predic_y,tf(k),Val_testX]=ENSVM_dual_osqp(Xtr,Ytr,Xt,FunPara);
    clear Predic_y
    Fk_Test(:,k)=Val_testX;
end
Tfinal=sum(tf);

for j=1:mt
    [max_fk,rk]=max(Fk_Test(j,:));
    clear max_fk
    Prediction(j,rk)=1;
end

[Loss,Bal_accu]=medidas_multi(Prediction,Yot,T);


