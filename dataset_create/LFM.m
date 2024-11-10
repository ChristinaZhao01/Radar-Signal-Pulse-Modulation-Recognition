function SNt = LFM(N,fmin,fmax,n) % 输入参数t为时间 返回加了噪声的10种dB类型的 信号
   SNt = fmlin(N,fmin,fmax);
   SNR = 2*n-12;
   SNt = awgn(SNt,SNR,'measured');
end
% A = 10;    % 幅值设置为1
% N = 1024; % 采样点数
% Fs = 200; % 采样频率 MHZ
% Ts = 1/Fs; % 采样周期
% T = N*Ts;  % 采样时间
% % t = (0:N-1)*T; % 时间序列
% B = 10+70*rand; % 信号带宽 10到80MHZ
% K = B/T; % 调频斜率
% f0 = 0; % 中心频率 MHZ
% SNR = 0; %  信噪比 dB
% 线性调频信号 s(t)=A*exp(li*2*pi(f0*t+K/2*t^2))
% temp = zeros(N,10);  % 储存不同SNR下的调制信号
% % St = A*exp(1j*(2*pi*f0*t + pi*K*t.^2)); % LFM的表达式 未加入噪声
% St = fmlin(N,fmin,fmax);
% k = 1; % 计次变量
% for SNR = -10 : 2 : 8
%     temp(:, k) = awgn(St,SNR,'measured');
%     k = k + 1;
% end
% SNt = temp;
% end

% % 3.LFM的模糊函数
% t = -T:T/N:T;
% f = -B:B/N:B;
% [tau,fd]=meshgrid(t,f); % 根据时间变量t 和频率变量f生成网络
% var1=T-abs(tau);
% var2=pi*(fd-K*tau).*var1;
% var2=var2+eps;
% amf=abs(sin(var2)./var2.*var1/T);
% amf=amf/max(max(amf));
% figure
% contour(tau.*1e6,fd*1e-6,amf);
% xlabel ('时延（us）');ylabel ('多普勒频率（MHz）');
% title('LFM信号等高图');
% grid on;axis tight;


