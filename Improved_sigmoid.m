clear;clc;
mm=[30,40,50,60];
n=80;
k=20;
mse = zeros(1,4);
MSE = zeros(3,4);%取三次重构mse的最小值
for time = 1:3
    for kk = 1:4
        m = mm(kk);
        out=zeros(1,n);
        A=normrnd(0,1,m,n);%产生(0,1)正太分布的系数矩阵A
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
    %     ox = awgn(ox,10);%加信噪比=10的高斯噪声
        [f_result,iteration,error]=rnn(ox,A,P,Q,I,m,n,f0);
        out=f_result(1:n,iteration-1);
        mse(kk) = sum((out-ox).^2)./(n^2);
    end
    mse;
    MSE(time,:) = mse;
end

function [df,count,error]=rnn(Original_value,A,P,Q,I,m,n,f0)
x(:,1)=f0(1:n);
iteration=150001;
step=0.01;
count=1;
error=zeros(1,iteration);
for i=1:iteration
%     dx=(2.0*exp(abs(x(:,i))).*sign(x(:,i)))./(exp(abs(x(:,i))) + 1.0).^2;%sigmoid delta=0.1
    dx = 10.*exp(-10.*abs(x(:,i))).*sign(x(:,i));%La delta=0.1
%     dx=(200*x(:,i))./(10000*x(:,i).^4 + 1).^(1/2) - (2000000*x(:,i).^5)./(10000*x(:,i).^4 + 1).^(3/2);%CT delta=0.1
%     dx = (2*x(:,i).^3)./(x(:,i).^2 + 1/100).^2 - (2*x(:,i))./(x(:,i).^2 +1/100); %comp delta=0.1
%     dx=200.0*x(:,i).*exp(-100.0*x(:,i).^2);%gau delta=0.1
    % dx = -(2.0.*(200.0.*x(:,i).*exp(-100.0*x(:,i).^2) - 200.0.*x(:,i).*exp(100.0*x(:,i).^2)))./(exp(-100.0.*x(:,i).^2) + exp(100.0*x(:,i).^2)).^2;%hyper delta=0.1
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
