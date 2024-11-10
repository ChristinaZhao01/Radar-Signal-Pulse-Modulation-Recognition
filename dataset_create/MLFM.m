function SNt = MLFM(N,n)
   [~,ifl1]=fmlin(N/2,0.125,0.25); [~,ifl2]=fmlin(N/2,0.2,0.245);
   iflaw = [ifl1;ifl2];
   SNt = fmodany(iflaw); 
   SNR = 2*n-12;
   SNt = awgn(SNt,SNR,'measured');
end

