%% Get Kernel

%% Add to path convolut library
addpath(genpath('/home/alexander/Desktop/WC_information/1_code/BioMultilayer_L_NL_convolut'));% BCN
%addpath(genpath('/media/disk/vista/Papers/WC_vs_DN/INCLUDING_deriv_f/integrability_WC/convergenceConvolut/BioMultilayer_L_NL_convolut'));% Valencia

%% SPATIAL SAMPLING FREQUENCY AND SPATIAL EXTENT OF IMAGE PATCHES 
% (this assumes certain viewing distance given size of display)

fs=64; % sampling frequency in cpd
% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment this in workstations)
% N=64;  % Number of samples -> 64*64 image blocks subtend 1 degree
% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment this in personal computers)
N= 40;  % Number of samples -> 64*64 image blocks subtend 1 degree (40*40 -> patches of less than 1 deg)

%% PARAMETERS OF THE 4 LAYERS OF THE MODEL
% (see description of the parameters in PARAMETERS_DN_ISO_COLOR.M)

% Layer 1: Brightness and RG-ness and YB-ness
brightness_type = 2;
g1=[1.25 1.25 1.25];  b1 = [0.01 0.01 0.01]; beta = [5 2 2]; mu = [0 0 0]; scale=[255 23 35]; % Weber-Laughlin-Malo-Bertalm
g1 = [0.7 0.7 0.7];

% Layer 2: Contrast computation (achromatic and chromatic)
Ls=0.066; Hs=0.066; Lc=1; Hc=1; b2=[30 10*23 10*35]; 
                                         % Gutierrez 12 + singul
% Layer 3: CSFs and masking in the spatial domain
obliq=1; lambda_regul_CSF=0.5e-4; b3=[0.005 0.00075 0.00075]; g3=1.5; H3s=0.02; H3c=1;        % Mullen-Barlow-Malo+reduct sing
% Layer 4: Wavelet transform and masking in the wavelet domain

% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment the following 4 lines in workstations -and comment the case below for small patches-)
%  ns = 4; no=4; tw=1; [p,ind] = buildSFpyr(rand(N,N),ns,no-1,tw); n_sub = length(ind(:,1));
% w = [50 50*[1 1 1 1] 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.001];                        % Malo-Bertalm-Wilson-Cowan
% c = [50 50*[1 1 1 1] 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.01];                         % 4 scales * 4 orient + 2 residuals
% b4 = 0.005*[1*[1 1 1 1 1] 1*[1 1 1 1] 10*[1 1 1 1] 100*[1 1 1 1] 0.8*100/2.5];
% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment the following 4 lines in personal computers -and comment the case above for big patches-)
ns = 3; no=4; tw=1; [p,ind] = buildSFpyr(rand(N,N),ns,no-1,tw); n_sub = length(ind(:,1));
w = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.001];
c = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.01];
b4 = 0.005*[1*[1 1 1 1 1] 10*[1 1 1 1] 100*[1 1 1 1] 0.8*100/2.5];
%%---------------
c_ff_interact = ones(n_sub,n_sub);                                                            
g4 = 0.6;
sigma_x = 0.02*ones(ns+2,1);  sigma_x(end)=0.01; sigma_scale = 1.1; sigma_orient = 30;        % Watson-Solomon
autonorm = 0; save_for_deriv = 1; save_for_inv = 1; save_achrom = 0;

load('alfa_b.mat')

tic
parameters_DN_iso_color_fast
toc



load('alfa_b.mat')

% Alpha modified
alfa_a_m = average_deviation_wavelet(alfa_m,param(4).ind);
param(4).kappa = kappa;
param(4).b = b;    % 30*
ga = 0.5;
Da = diag(alfa_m);


%% Generate kernel to export 

Wpy = {}
for fo = 1:size(param(4).ind,1)
    for fo2 = 1:size(param(4).ind,1)
        Wpy{fo,fo2} = rot90(param(4).Hxz{fo,fo2}*param(4).Hsz(fo,fo2)*param(4).Hoz(fo,fo2),2);
    end
end

W = Wpy;

save('W','W')
save('alfa_m','alfa_m')
