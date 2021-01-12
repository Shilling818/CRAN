function output = GaussianLF(input, n, D0)
    %���ð�����˹��ͨ�˲�����ͼ������˲�
    M = 2 * size(input, 1);  %�˲�������
    N = 2 * size(input, 2);  %�˲�������
    u = - M/2:(M/2 - 1);
    v = - N/2:(N/2 - 1);
    [U,V] = meshgrid(u,v);
    D = sqrt(U.^2 + V.^2);
%     D0 = 50;
%     n = 6;
%     H = 1./(1+(D./D0).^(2*n));  
    H = exp(- (D./ D0).^n / 2);
    J = fftshift(fft2(input, size(H,1),size(H,2)));  %ת����Ƶ��
    K = J.*H;
    L = ifft2(ifftshift(K));  %����Ҷ���任
    L = L(1:size(input,1), 1:size(input, 2), :);  %�ı�ͼ���С
    output = L;
end