function SNt = BPSK(N,n)
   [SNt,~]=anabpsk(N,70,0.25); 
   SNR = 2*n-12;    %步进为2db的从-10到8db的信噪比
   SNt = awgn(SNt,SNR,'measured');     
end

