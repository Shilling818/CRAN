function [mssim, ssim_map,siga_sq,sigb_sq] = SSIM(ima, imb)  
% ========================================================================  
%ssim的算法主要参考如下论文：  
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image  
% quality assessment: From error visibility to structural similarity,"  
% IEEE Transactios on Image Processing, vol. 13, no. 4, pp. 600-612,  
% Apr. 2004.  
%  首先对图像加窗处理，w=fspecial('gaussian', 11, 1.5);  
%                 (2*ua*ub+C1)*(2*sigmaa*sigmab+C2)  
%   SSIM(A,B)=————————————————————————  
%              (ua*ua+ub*ub+C1)(sigmaa*sigmaa+sigmab*sigmab+C2)  
%     C1=（K1*L）;  
%     C2=(K2*L);   K1=0.01，K2=0.03  
%     L为灰度级数，L=255  
%-------------------------------------------------------------------  
%     ima - 比较图像A  
%     imb - 比较图像B  
%  
% ssim_map - 各加窗后得到的SSIM（A,B|w）组成的映射矩阵  
%    mssim - 对加窗得到的SSIM（A,B|w）求平均，即最终的SSIM（A,B）  
%  siga_sq - 图像A各窗口内灰度值的方差  
%  sigb_sq - 图像B各窗口内灰度值的方差  
%-------------------------------------------------------------------  
%  Cool_ben  
%========================================================================  
  
w = fspecial('gaussian', 11, 1.5);  %window 加窗  
K(1) = 0.01;                      
K(2) = 0.03;                      
L = 255;       
ima = double(ima);  
imb = double(imb);  
  
C1 = (K(1)*L)^2;  
C2 = (K(2)*L)^2;  
w = w/sum(sum(w));  
  
ua   = filter2(w, ima, 'valid');%对窗口内并没有进行平均处理，而是与高斯卷积，  
ub   = filter2(w, imb, 'valid'); % 类似加权平均  
ua_sq = ua.*ua;  
ub_sq = ub.*ub;  
ua_ub = ua.*ub;  
siga_sq = filter2(w, ima.*ima, 'valid') - ua_sq;  
sigb_sq = filter2(w, imb.*imb, 'valid') - ub_sq;  
sigab = filter2(w, ima.*imb, 'valid') - ua_ub;  
  
ssim_map = ((2*ua_ub + C1).*(2*sigab + C2))./((ua_sq + ub_sq + C1).*(siga_sq + sigb_sq + C2));  
  
mssim = mean2(ssim_map);  
  
return  