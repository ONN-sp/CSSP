close all;                        % 清理工作空间
clear;  
tic
[I_noise,map] = imread('');
I_noise=rgb2gray(I_noise);
[LoD,HiD] = wfilters('bior3.7','d');%小波滤波器
[cA,cH,cV,cD] = dwt2(I_noise,LoD,HiD,'mode','symh');
figure, imshow(I_noise);
m=240;
n=400;%让dwt后的近似矩阵稀疏
k=size(cA,1);
cA=round(cA./10);
ox=zeros(k,n);
out=zeros(k,n);
final_out=zeros(k,k);
index_all=zeros(k,k);
A=normrnd(0,1,m,n);%产生(0,1)正太分布的系数矩阵A
mse = zeros(1,k);
for i=1:k
index=randperm(n,k);%返回一行从1到n的整数中的k个，而且这k个数也是不相同的
index_all(i,:)=index;
for count=1:k
  ox(i,index(count))=cA(i,count);
end
    b=A*ox(i,:)';
    tend=0.01;
    x=rand(n,1);%生成n*1的随机矩阵
    f0=x;
    P=A'*inv(A*A')*A;
    [mp,np]=size(P);
    I=eye(mp);
    Q=A'*inv(A*A')*b;
    [f_result,iteration]=rnn(A,P,Q,I,m,n,f0);
%     [ox(i,:)',f_result(1:n,iteration-1)];
     out(i,:)=f_result(1:n,iteration-1);
     for kk=1:k
     final_out(i,kk)=out(i,index(kk));
     end
toc
end
mse = sum((final_out-cA).^2)./(n^2);
MSE = sum(mse);
PSNR = 10*log10(255^2/MSE);
error = norm(final_out-cA,2)/norm(cA,2);
figure(3)
Y=idwt2(round(final_out)*10,cH,cV,cD,'bior3.7');%单尺度二维离散小波重构(逆变换)
imshow(Y,map)
function [df,count]=rnn(A,P,Q,I,m,n,f0)
iteration=60001;
x(:,1)=f0(1:n);
step=0.01;
count=1;
for i=1:iteration
    dx=(2.0*exp(abs(x(:,i))).*sign(x(:,i)))./(exp(abs(x(:,i))) + 1.0).^2;%sigmoid delta=0.1
    x(:,i+1)=x(:,i)+step*(-P*x(:,i)-(I-P)*dx+Q);
    xxx(:,count)=x(:,i+1);
    count=count+1;
end
df=xxx;
end





