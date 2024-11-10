function SNt = FSK2(N,n,m)  % n跟信噪比相关 m跟次数相关 2 4
   [SNt,~]=anafsk(N,200,m); % 2FSK
    SNR = 2*n-12;
    SNt = awgn(SNt,SNR,'measured');
end

