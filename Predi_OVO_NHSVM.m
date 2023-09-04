function [Loss,Bal_accu,Tfinal]=Predi_OVO_NHSVM(Xt,Yot,Xa,Ya,FunPara,T)

mt=size(Xt,1);
Prediction=-ones(mt,T);
tf=zeros(T);
cont_k=zeros(T,mt);
for i=1:T
    fin1=find(Ya==i);
    DataTrain.A=Xa(fin1,:);
    for j=i+1:T
        fin2=find(Ya==j);
        DataTrain.B=Xa(fin2,:);
        [Predict_Y,w1,w2,b1,b2,tf(i,j)]=NSVM(Xt,DataTrain,FunPara);     
        clear w1 w2 b1 b2 
        for kl=1:mt
            if Predict_Y(kl)==1
               cont_k(i,kl)=cont_k(i,kl)+1;
            else
               cont_k(j,kl)=cont_k(j,kl)+1; 
            end
        end
        
    end
end
Tfinal=sum(sum(tf));
for j=1:mt 
    [max_vot,r]=max(cont_k(:,j));
    Prediction(j,r)=1;
end

[Loss,Bal_accu]=medidas_multi(Prediction,Yot,T);
% Loss=sum(sum(Yot~= Prediction))/(2*size(Yot,1)); 
% for k=1:T
%     xi=Prediction(:,k); % Prediccion
%     yi=Yot(:,k);   % Etiqueta real
%     if sum(yi==1)>0
%         balan(k)= sum(xi==1  & yi==1)/sum(yi==1);
%     else
%         balan(k)=1;
%     end
% end
% bal_accu=mean(balan);
