clear;clc;
x=-1:0.01:1;
y2=1-2./(1+exp(abs(x)/0.1^2));
y3 = 1-exp(-(x.^2)./(2*0.1^2));
y4 = 1-exp(-abs(x)/0.1);
y5 = sin(atan(x.^2/0.1^2));
y6 = x.^2./(x.^2+0.1^2);
y7 = 1-2*(exp(-x.^2/0.1^2)+exp(x.^2/0.1^2)).^(-1);
plot(x,y2,'r','LineWidth',2)
hold on
plot(x,y3,'b','LineWidth',2)
hold on
plot(x,y4,'k','LineWidth',2)
hold on
plot(x,y5,'m','LineWidth',2)
hold on
plot(x,y6,'c','LineWidth',2)
hold on
plot(x,y7,'g','LineWidth',2)
hold on
xlabel('x')
ylabel('f(x)')
legend('Sum.improved.sigmoid','Sum.inv.Gaussian','Sum.inv.Laplacian','Sum.symmetric.CT','Sum.comp.inv.func','Sum.inv.hyper','loctaion')