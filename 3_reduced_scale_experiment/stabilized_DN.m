function [y1,x1,y2,x2,J1,J2,param] = stabilized_DN(x0,L1,g1,S2,g2,b2,H2,left,right,e_star,kappa,calcula)

%
%  STABILIZED_DN is a two-layer L+NL model applicable to 3-pixel images.
%  For a given input column vector x0 this routine computes the transform and the Jacobian wrt input
%
%    First Layer: point-wise brightness
%
%           y1 = L1*x0
%           x1 = y1^g1
%
%    Second Layer: stabilized Fourier transform and Divisive Normalization
%
%           y2 = S2*F*x1
%           x2 = K(e*)*sign(y2)*(e/(b+H*e));      where e = y2^g2 and K(e*) is fixed and computed from natural images 
%
%  Example parameters:
%  -------------------
% 
%         L1 = eye(3);                                                                    
%         g1= 0.7;
%         % F2 = Fourier transform (DC,low,high) = [1 1 1;1 0 -1;-0.5 1 -0.5]  
%         S2 = diag([1 0.5 0.2]); 
%         g2 = 0.7;
%         H2 = [1 0.3 0.1;0.3 1 0.3;0.1 0.3 1];
%         b2 = 0.1*[1;0.5;0.2];
%
%  [y1,x1,y2,x2,J1,J2,param] = stabilized_DN(x0,L1,g1,S2,g2,b2,H2norm,f_left,f_right,e_star,kappa,calculaJ);
%

param(1).general = 1;
param(1).s_wavelet = 0;
param(1).L = L1;
param(1).iL = inv(param(1).L);
param(1).g = g1;
param(1).b = ones(3,1);
param(1).H = 0;
param(1).d = 3;
param(1).Hc = 0;
param(1).Hw = 0;

param(2).general = 1;
param(2).s_wavelet = 0;
L = [1 1 1;1 0 -1;-0.5 1 -0.5];          % LINEAR a: (Normalized) frequency filters
L = L./repmat(sqrt(sum(L.^2,2)),1,3);    
%S = diag([1 0.5 0.2]);
param(2).L = S2*L;
param(2).iL = inv(param(2).L);
param(2).g = g2;
param(2).b = b2; % 0.1*[1;0.5;0.2];
% H = [1 0.3 0.1;0.3 1 0.3;0.1 0.3 1];
H2 = H2./repmat(sum(H2,2),1,3);
H = diag(left)*H2*diag(right);
param(2).H = H;
param(2).d = 3;
param(2).Hc = left;
param(2).Hw = right;
param(2).kappa = kappa;
param(2).e_star = e_star;

K = kappa.*((b2+H*e_star)./e_star);

param(2).K = K;

if calcula==1
   comp_J.sx=1;
else
   comp_J=0; 
end

[y1,x1,J1] = stage_L_NL_c(x0,param(1),comp_J);
[y2,x2,J2] = stage_L_NL_c(x1,param(2),comp_J);

x2 = diag(K)*x2;

if calcula==1
   J2.sx = diag(K)*(J2.sx);
end