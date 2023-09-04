function [Loss,bal_accu,Tfinal]=Predi_OVA_NHSVM_osqp(Xt,Yot,Xa,Ya,FunPara,T)
mt=size(Xt,1);
Prediction1=-ones(mt,T);
Prediction2=-ones(mt,T);
Prediction3=-ones(mt,T);
tf=zeros(T,1);

for k=1:T
    fin1=find(Ya==k); 
    fin2=setdiff([1:length(Ya)]',[fin1]);
    DataTrain.A=Xa(fin1,:);
    DataTrain.B=Xa(fin2,:);
    [Predic_y1,Predic_y2,Predic_y3,tf(k)]=NSVM_M_osqp(Xt,DataTrain,FunPara);
    Fk_Test1(:,k)=abs(Predic_y1);
    Fk_Test2(:,k)=abs(Predic_y2);
    Fk_Test3(:,k)=Predic_y3;
end
Tfinal=sum(tf);

for j=1:mt
    [min_fk,rk1]=min(Fk_Test1(j,:));
    [max_fk,rk2]=max(Fk_Test2(j,:));
    [max_fk,rk3]=max(Fk_Test3(j,:));
    Prediction1(j,rk1)=1;
    Prediction2(j,rk2)=1;
    Prediction3(j,rk3)=1;
end

[Loss(1),bal_accu(1)]=medidas_multi(Prediction1,Yot,T);
[Loss(2),bal_accu(2)]=medidas_multi(Prediction2,Yot,T);
[Loss(3),bal_accu(3)]=medidas_multi(Prediction3,Yot,T);


