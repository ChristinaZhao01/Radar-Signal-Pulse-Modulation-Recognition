clc,clear,close all; 
% 生成实验五的数据集 12种调制信号加噪声 详细参数见论文
% 设置参数
N = 1024; % 采样点数
fs = 200; % 采样频率 MHZ
Ts = 1/fs; % 采样时间
B = 10+70*rand;%带宽在10~80Mhz
T = N*Ts;
K = B/T; 

% Frank调制
%for ii = 1 : 10     %信噪比10个步进
%for jj = 1 : 800    %每种信噪比信号生成800张图
%SNt = Frank(N,ii);  %生成不同信噪比的调制信号，步进为2db
%g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
%t=1:1024; 
%tfrcw(SNt,t,1024,g,h,1.8,0);
%set(gcf,'Position',[0,0,256,256]);
%set(gca,'Position',[0,0,1,1]);		%去除白边
%if jj > 700     %生成700张作为训练集
%    axis xy;axis off;colormap jet;
%    saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\Frank\test',num2str(ii),'\'),num2str(jj-700),'.png']);
%else            %生成100张作为测试集
%    axis xy;axis off;colormap jet;
%    saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\Frank\train',num2str(ii),'\'),num2str(jj),'.png']);
%end
%axis off;  %关闭坐标
%end
%end
%fprintf("picture is ok")
% FSKBPSK 调制
 for ii = 1 : 10        %信噪比10个步进
 for jj = 1 : 800       %每种信噪比信号生成800张图
 SNt = FSKBPSK(N,ii);   %生成不同信噪比的调制信号，步进为2db
 g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
 t=1:1024; 
 tfrcw(SNt,t,1024,g,h,1.8,0);       %进行时频变换
 set(gcf,'Position',[0,0,256,256]);
 set(gca,'Position',[0,0,1,1]);		%去除白边
 if jj > 700
     axis xy;axis off;colormap jet;
     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\FSKBPSK\test',num2str(ii),'\'),num2str(jj-700),'.png']);
 else
     axis xy;axis off;colormap jet;
     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\FSKBPSK\train',num2str(ii),'\'),num2str(jj),'.png']);
 end
 axis off;  %关闭坐标
 end
 end
 fprintf("picture is ok");
%% MP 调制
% for ii = 5 : 10
% for jj = 1 : 800
% SNt = LFM(N,0.15,0.15,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\MP\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\MP\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% end
% end
% fprintf("picture is ok");
%% LFMBPSK 连调
% for ii = 1 : 10
% for jj = 1 : 800
% SNt = LFMBPSK(N,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\LFMBPSK\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\LFMBPSK\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% end
% end
% fprintf("picture is ok");
%% MLFM 错开的三角形
% for ii = 1 : 10
% for jj = 1 : 800
% SNt = MLFM(N,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\MLFM\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\MLFM\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% end
% end
% fprintf("picture is ok");
 %% DLFM 三角形频率调制
% for ii =  10
% for jj = 1 : 800
% SNt = DLFM(N,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\DLFM\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\DLFM\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% end
% end
% fprintf("picture is ok");
 %% EQFM 二次偶调制 
% for ii = 1 : 10
% for jj = 1 : 800
% SNt = EQFM(N,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\EQFM\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\EQFM\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% % close(h1);
% end
% end
% fprintf("picture is ok");
 %% FSK调制信号  2FSK和4FSK
% for kk = 2 : 2 : 4  % 俩种 2FSK  4FSK
% for ii = 1 : 10
% for jj = 1 : 800
% SNt = FSK2(N,ii,kk);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\',num2str(kk),'FSK\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\',num2str(kk),'FSK\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% % close(h1);
% end
% end
% end
% fprintf("picture is ok");
 %% BPSK调制信号
% for ii = 1 : 10
% for jj = 1 : 800
% fmin = 0.1;
% fmax = fmin+0.3;
% SNt = BPSK(N,ii);
% g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
% % h1 = figure(1);
% t=1:1024; 
% tfrcw(SNt,t,1024,g,h,1.8,0);
% set(gcf,'Position',[0,0,256,256]);
% set(gca,'Position',[0,0,1,1]);		%去除白边
% if jj > 700
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\BPSK\test',num2str(ii),'\'),num2str(jj-700),'.png']);
% else
%     axis xy;axis off;colormap jet;
%     saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\BPSK\train',num2str(ii),'\'),num2str(jj),'.png']);
% end
% axis off;  %关闭坐标
% % close(h1);
% end
% end
% fprintf("picture is ok");


% 线性调制信号 LFM
for ii = 1 : 10
for jj = 1 : 800
fmin = 0.1;
fmax = fmin+0.3;
SNt = LFM(N,fmin,fmax,ii);
g=tftb_window(31,'Kaiser'); h=tftb_window(107,'Kaiser'); 
h1 = figure(1);
t=1:1024; 
tfrcw(SNt,t,1024,g,h,1.8,0);
set(gcf,'Position',[0,0,256,256]);
set(gca,'Position',[0,0,1,1]);		
if jj > 700
    saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\LFM\test',num2str(ii),'\'),num2str(jj-700),'.png']);
else
    saveas(gcf,[strcat('D:\python work\Py work folder\data\picture\LFM\train',num2str(ii),'\'),num2str(jj),'.png']);
end
axis off;  %关闭坐标
close(h1);
end
end
fprintf("picture is ok");

	











