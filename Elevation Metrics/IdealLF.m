function output = IdealLF(input, D0)
    %���ð�����˹��ͨ�˲�����ͼ������˲�
    M = 2 * size(input, 1);  %�˲�������
    N = 2 * size(input, 2);  %�˲�������
    u = - M/2:(M/2 - 1);
    v = - N/2:(N/2 - 1);
    [U,V] = meshgrid(u,v);
    D = sqrt(U.^2 + V.^2);
    H = zeros(size(D,1),size(D,2));
    H(D <= D0) = 1;
    J = fftshift(fft2(input, size(H,1),size(H,2)));  %ת����Ƶ��
    K = J.*H;
    L = ifft2(ifftshift(K));  %����Ҷ���任
    L = L(1:size(input,1), 1:size(input, 2), :);  %�ı�ͼ���С
    output = L;
end