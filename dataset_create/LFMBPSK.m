function SNt = LFMBPSK(N,n)
   fmin = 0;
   fmax = 0.2;
   SNt = anabpsk(N,150,0.1).*fmlin(N,fmin,fmax);
   SNR = 2*n-12;
   SNt = awgn(SNt,SNR,'measured');
end

