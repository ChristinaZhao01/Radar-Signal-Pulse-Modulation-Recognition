function SNt = FSKBPSK(N,n)
    SNt = anafsk(N,200,2).*anabpsk(N,70,0.25);
    SNR = 2*n-12;
    SNt = awgn(SNt,SNR,'measured');
end

