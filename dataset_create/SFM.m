function SNt = SFM(N,fmin,fmax,n)
SNt = fmsin(N,fmin,fmax,1024,512,(fmax+fmin)/2,-1.0);
SNR = 2*n-12;
SNt = awgn(SNt,SNR,'measured');
end


