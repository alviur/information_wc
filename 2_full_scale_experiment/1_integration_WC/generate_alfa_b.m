%% Paths
h = genpath('/home/alexander/Desktop/WC_information/1_code/BioMultiLayer_L_NL_color');
addpath(h)




%-------------------------------------------------------------------------
% DN MODEL PARAMETERS (color)
%-------------------------------------------------------------------------

   fs=64; N=40;
   % Stage 1: where parameters for brightness and chromatic channels may be different and controlled here
   g1=[1.25 1.25 1.25];  b1 = [0.01 0.01 0.01]; beta = [5 2 2]; mu = [0 0 0]; scale=[260 23 35];
   brightness_type = 1;
   % Stage 2 (where parameters for achromatic and chromatic are the same except for the semisaturation (relatively quite big for the chromatic channels)
   Ls=0.066; Hs=0.066; Lc=1; Hc=1; b2=[30 10*23 10*35];
   % Stage 3 same parameters except for the CSFs (proper CSFs are computed inside parameters_DN_iso_color)
   obliq=1; lambda_regul_CSF=0.5e-4; b3=[0.005 0.00075 0.00075]; g3=1.5; H3s=0.02; H3c=1; 
   % Stage 4: Same parameters for achromatic and chromatic sensors
   ns = 3; no=4; tw=1; [p,ind] = buildSFpyr(rand(N,N),ns,no-1,tw); n_sub = length(ind(:,1)); 
   c_ff_interact = ones(n_sub,n_sub); % 4 scales * 4 orient + 2 residuals
   w = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.001];                              % Malo-Bertalm-Wilson-Cowan
   c = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.01];
   b4 = 0.005*[1*[1 1 1 1 1] 10*[1 1 1 1] 100*[1 1 1 1] 0.8*100/2.5];
   g4 = 0.6;
   sigma_x = 0.02*ones(ns+2,1);  sigma_x(end)=0.02; sigma_scale = 1.1; sigma_orient = 30; % Watson-Solomon
   autonorm = 0; save_for_deriv = 1; save_for_inv = 1; save_achrom = 0;
      
   tic
   parameters_DN_iso_color
   toc
   
% -----------------------------
% Wilson-Cowan parameters
% -----------------------------
   
   W = (param(4).Hx).*(param(4).Hsc).*(param(4).Ho);

%% Alpha original
kappa = param(4).kappa(:,1);
[Structure_Matrix , b] = structure( param(4).ind , param(4).d , param(4).b);
K = kappa.*(b + param(4).H*param(4).e_average)./param(4).e_average;
alfa = b./K;
alfa_a = average_deviation_wavelet(alfa,param(4).ind);

figure(1),plot(b),title('b original')
figure(2),plot(K),title('K original')
figure(3),plot(alfa),title('\alpha original')

%% Alpha modified
kappa = param(4).kappa;
kappa = 0.1*kappa;
kappa(1:8000,:) = 0.001*kappa(1:8000,:);
kappa(8001:9600,:) = 0.7*kappa(8001:9600,:);
[Structure_Matrix , b] = structure( param(4).ind , param(4).d , param(4).b);
b(1:8000) = 30*b(1:8000);
b(8001:9600) = 4*b(8001:9600);
K = kappa(:,1).*(b + param(4).H*param(4).e_average)./param(4).e_average;
alfa_m = b./K;
alfa_a_m = average_deviation_wavelet(alfa_m,param(4).ind);
param(4).kappa = kappa;
param(4).b(1:5) = 30*param(4).b(1:5);
param(4).b(6:9) = 4*param(4).b(6:9);

b = param(4).b;

save('alfa_b.mat','alfa_m','b','kappa')