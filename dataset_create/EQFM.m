function SNt = EQFM(N,n)
   [SNt,~]=fmpar(N,[1 0.32],[512 0.05],[1024 0.32]);
   SNR = 2*n-12;
   SNt = awgn(SNt,SNR,'measured');
end

