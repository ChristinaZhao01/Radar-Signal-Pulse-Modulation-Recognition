function SNt = Frank(NN,nn) % N为4*4 NN为1024
   N = 16;
   show = zeros(N,1);
   k = 1;
   M = sqrt(N);
   for i=1:M
       for j = 1:M
           show(k,1) = (i-1)*(j-1)/M*2*pi;
           k = k+1;
       end
   end
code = show;
fy=NN;%采样频率
XHCD=50;
MYSLpsk=50;
t =1/fy:1/fy:(XHCD*(1/MYSLpsk));%码元长度*每个码元的持续时间=总时间
t =t*fy;
h=code*pi;
m=zeros(1,length(t));
time_duration=round(length(t)/XHCD);
dis = length(h);
for j=1:XHCD
    if mod(j,dis) == 0
        m((j-1)*time_duration+1:(j)*time_duration)=h(dis);
    else
        m((j-1)*time_duration+1:(j)*time_duration)=h(mod(j,dis));
    end
end
m=m(1:length(t));
fcpsk = 0.25;
SNt =exp(1i*(2*pi*(fcpsk)*t+m));%调制后的信号mpsk,其中每个码元的相位叠加不一样
SNt = SNt';
SNt = SNt.*fmlin(1024,0,0.01);
SNR = 2*nn-12;
SNt = awgn(SNt,SNR,'measured');
% XHCD=100;
% MYSLpsk=100;
% t =1/fy:1/fy:(XHCD*(1/MYSLpsk));%码元长度*每个码元的持续时间=总时间
end
