clear;clc;
m=60;
n=80;
k=20;

out=zeros(1,n);
A=normrnd(0,1,m,n);%产生(0,1)正太分布的系数矩阵A
for i=1:n
    k_v=-2*unidrnd(5,k,1)+unidrnd(5,k,1);%unidrnd生成(连续)均匀分布的随机整数,返回的是k*1矩阵。这里产生的其实是非零元
    k_v(k_v==0)=1;
    ox=zeros(n,1);
    index=randperm(n,k);%返回一行从1到n的整数中的k个，而且这k个数也是不相同的
    for count=1:k
        ox(index(count))=k_v(count);
    end
    b=A*ox; 
    tend=0.01;
    x=rand(n,1);%生成n*1的随机矩阵
    f0=x;
    P=A'*inv(A*A')*A;
    [mp,np]=size(P);
    I=eye(mp);
    Q=A'*inv(A*A')*b;
    [f_result,iteration,error]=rnn(ox,A,P,Q,I,m,n,f0);
    out(i,:)=f_result(1:n,iteration-1);
end

figure(8)
semilogy(1:iteration-1,error)
axis([0,3000,0,1])
xlabel('Iteration')
ylabel('Relative error \epsilon(log)')


function [df,count,error]=rnn(Original_value,A,P,Q,I,m,n,f0)
x(:,1)=f0(1:n);
iteration=300001;
step=0.01;
count=1;
error=zeros(1,iteration);
for i=1:iteration
    dx=(12.73*x(:,i))./(100*x(:,i).^4 + 1.0);%arc delta=0.1
    x(:,i+1)=x(:,i)+step*(-P*x(:,i)-(I-P)*dx+Q);
    error(i)=sqrt(sum(abs(x(:,i+1)-Original_value).^2)/sum(abs(x(:,i+1)).^2));%每一次迭代恢复的向量与原向量的误差值
    xxx(:,count)=x(:,i+1);
    count=count+1;
    if error(i)<10^-7
        break
    end
end
df=xxx;
end
