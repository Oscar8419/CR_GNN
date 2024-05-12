
% 仿真名称： Laplace噪声下噪声不确定性对性能的影响
% 仿真条件： SNR=-10dB，N=200，p=0.4，p=0.6，p=0.8，b=1/sqrt(2)，U=0，4，6。
clear all;
clc;
snr=-10;
vars=10^(snr/10);
N=10;   %采样十个时刻
M=20;   %采样二十个节点
MN=200;
p1=1;%%阶数p的取值
p2=0.1;
p3=0.1;
p4=0.6;
p5=0.8;
rho=0.9;%%相关系数
for i=1:MN  %
    mu=0;                      %均值
    sigma=1;                  %标准差，方差的开平方
    b=sigma/sqrt(2);      %根据标准差求相应的b
    a=rand(M,N)-0.5;
    x=mu-b*sign(a).*log(1-2*abs(a)); %%生成符合拉普拉斯分布的随机数列
    q1=sqrt(vars)*randn(M,N);
    for m=1:M  
        for k=1:N
            if(k==1)
                z(m,k)=q1(m,k);
            else
                z(m,k)=rho*z(m,k-1)+q1(m,k);
            end
        end

    end
    %%%%%%***  p1 ****
    r1=(x.^p1)*(x.^p1)'/N;   %times conj-transpose
    %信号加噪声
    z2=z+x;
    z3=(z2.^p1)*(z2.^p1)'/N;

    noise = diag(r1);
    signal = diag(z3);

end


save('data.mat', 'noise', 'signal');