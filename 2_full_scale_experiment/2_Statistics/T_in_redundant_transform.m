
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

figure(4),plot(b),title('b modified')
figure(5),plot(K),title('K modified')
figure(6),plot(alfa_m),title('\alpha modified')

%%
%%   Wilson-Cowan kernel
%%

%pix_to_rem = [4 4 4 4 4 2 2 2 2 1 1 1 1 1];
%[W_focus, ind_focus] = focus_on_center_kernel(W,param(4).ind,pix_to_rem);
%position = pos_s_pyr([2 1 7 7],ind_focus);
%W_coef_focus = W_focus(position,:);
%figure,showSpyr(W_coef_focus,ind_focus);

position = pos_s_pyr([2 1 10 10],param(4).ind);
W_coef = W(position,:);
pix_to_rem = [4 4 4 4 4 3 3 3 3 1 1 1 1 1];
[W_coef_focus,ind_focus] = focus_on_center(W_coef',param(4).ind,pix_to_rem);

figure,showSpyr(W_coef,param(4).ind);
figure,showSpyr(W_coef_focus,ind_focus);

pix_to_rem_jesus = [4 4 4 4 4 2 2 2 2 1 1 1 1 1];
[W_focus_jesus, ind_focus_jesus] = focus_on_center_kernel(W,param(4).ind,pix_to_rem_jesus);


%%-------------------------
%%   Lets analyze the wavelet transform
%%-------------------------

%%%%%%%%%%%%%%% Theory 1: extra total correlation according to Studeny 
%%%%%%%%%%%%%%%           (including the generalization for non-rectangular jacobians)

Lw = param(4).L;
LwTLw = Lw'*Lw;

tic,[B,lambda_lwtlw] = eig(LwTLw);toc
E_log_det_LwTLw_05 = 0.5*sum(log(diag(lambda_lwtlw)));

%%%%%%%%%%%%%%% Theory 2: extra total correlation according to the first term of Cardoso 
%%%%%%%%%%%%%%%           (negentropy is invariant under affine and marginal negentropy of Gaussian noise transformed is zero)
%%%%%%%%%%%%%%%            -> extra total correlation comes from the new
%%%%%%%%%%%%%%%            covariance, which, assuming uncorrelated Gaussian input, 
%%%%%%%%%%%%%%%            the covariance at the output is Cw = Lw*Lw'
%%%%%%%%%%%%%%%            Cardoso's term is log(|diag(Cw)|)/log(|Cw|)

Cw = Lw*Lw';
tic,[B,lambda_Cw] = eig(Cw);toc

C_cardoso = 0.5 * (sum(log(diag(Cw))) - sum(log(diag(lambda_Cw))));

%%clear param
%save parameters_WC_for_theoretical_TotCorr

%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%  Empirical Analysis of Noise
%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%

nw = Lw*randn(1600,5175);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%   NOISE SAMPLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%  SELECCION 1/3: TAMANYOS WAVELET y ESPACIAL
%%%

tam_low_select = 3;     % 5    mas lejanos  7
tam_med_select = 6;     % 10                14 
tam_hig_select = 12;    % 20                28
tam_space_select = 12;  % 20                28 

%%%
%%%  SELECCION WAVELET
%%%

% %%%%%% Seleccion optimista (18 dimensiones en low y med)
% %%%%%% 
% 
% % seleccion low
% tam_low = tam_low_select;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d 
% 
% % Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_med = tam_med_select;
% tam_low = tam_med;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d
% 
% % Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_high = tam_hig_select;
% tam_low = tam_high;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (10 dimensiones en low y med)
%%%%%%

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high_cons= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 1;
samples_nw_focus = [];
for i = 1:length(nw(1,:))
    [1 i]
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    samples = scan_wavelet_for_spatial_samples(nw_focus,ind_focus,tam_low_select,conservative);
    samples_nw_focus = [samples_nw_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_12_40_noise_low_dim samples_nw_focus -v7.3

%%%
%%%  SELECCION 2/3: TAMANYOS WAVELET y ESPACIAL
%%%
clear samples_nw_focus

tam_low_select = 5;     % 5    mas lejanos  7
tam_med_select = 10;     % 10                14 
tam_hig_select = 20;    % 20                28
tam_space_select = 20;  % 20                28 

%%%
%%%  SELECCION WAVELET
%%%

% %%%%%% Seleccion optimista (18 dimensiones en low y med)
% %%%%%% 
% 
% % seleccion low
% tam_low = tam_low_select;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d 
% 
% % Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_med = tam_med_select;
% tam_low = tam_med;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d
% 
% % Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_high = tam_hig_select;
% tam_low = tam_high;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (10 dimensiones en low y med)
%%%%%%

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high_cons= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 1;
samples_nw_focus = [];
for i = 1:length(nw(1,:))
    [1 i]
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    samples = scan_wavelet_for_spatial_samples(nw_focus,ind_focus,tam_low_select,conservative);
    samples_nw_focus = [samples_nw_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_20_40_noise_low_dim samples_nw_focus -v7.3


%%%
%%%  SELECCION 3/3: TAMANYOS WAVELET y ESPACIAL
%%%
clear samples_nw_focus

tam_low_select = 7;     % 5    mas lejanos  7
tam_med_select = 14;     % 10                14 
tam_hig_select = 28;    % 20                28
tam_space_select = 28;  % 20                28 

%%%
%%%  SELECCION WAVELET
%%%

% %%%%%% Seleccion optimista (18 dimensiones en low y med)
% %%%%%% 
% 
% % seleccion low
% tam_low = tam_low_select;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d 
% 
% % Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_med = tam_med_select;
% tam_low = tam_med;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d
% 
% % Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
% tam_high = tam_hig_select;
% tam_low = tam_high;
% indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
% cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
%                    1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
% cuales_low_2d = zeros(tam_low,tam_low);
% for i = 1:length(cuales_low_list)             
%     cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
% end
% seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% % coincide con: 
% cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (10 dimensiones en low y med)
%%%%%%

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high_cons= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 1;
samples_nw_focus = [];
for i = 1:length(nw(1,:))
    [1 i]
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    samples = scan_wavelet_for_spatial_samples(nw_focus,ind_focus,tam_low_select,conservative);
    samples_nw_focus = [samples_nw_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_28_40_noise_low_dim samples_nw_focus -v7.3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (2) ..... 961 por imagen -> 4973175
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load marginal_statistics_foster

tam_low = 2;     % 3, 4
tam_space = 10;  % 13, 20
tam_por_imagen = 961;
tam_total = tam_por_imagen*5175;

w = nw;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(nw(1,:))
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(nw_focus,ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [1 i] 
    end
end
    s_H1_n4=s_H1s;    s_H2_n4=s_H2s;    s_H3_n4=s_H3s;    s_H4_n4=s_H4s;
    s_M1_n4=s_M1s;    s_M2_n4=s_M2s;    s_M3_n4=s_M3s;    s_M4_n4=s_M4s;
    s_L1_n4=s_L1s;    s_L2_n4=s_L2s;    s_L3_n4=s_L3s;    s_L4_n4=s_L4s;
    s_lows_n4=s_lows;
    
save samples_spatial_wav_tam_2_noise_full ...
s_H1_n4 s_H2_n4 s_H3_n4 s_H4_n4 s_M1_n4 s_M2_n4 s_M3_n4 s_M4_n4 s_L1_n4 s_L2_n4 s_L3_n4 s_L4_n4 s_lows_n4 -v7.3

%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (3)  .... 784 -> 4057200
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load marginal_statistics_foster

tam_low = 3;     % 3, 4
tam_space = 13;  % 13, 20
tam_por_imagen = 784;
tam_total = tam_por_imagen*5175;

w = nw;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(nw(1,:))
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(nw_focus,ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [1 i] 
    end
end
    s_H1_n4=s_H1s;    s_H2_n4=s_H2s;    s_H3_n4=s_H3s;    s_H4_n4=s_H4s;
    s_M1_n4=s_M1s;    s_M2_n4=s_M2s;    s_M3_n4=s_M3s;    s_M4_n4=s_M4s;
    s_L1_n4=s_L1s;    s_L2_n4=s_L2s;    s_L3_n4=s_L3s;    s_L4_n4=s_L4s;
    s_lows_n4=s_lows;
    
save samples_spatial_wav_tam_3_noise_full ...
s_H1_n4 s_H2_n4 s_H3_n4 s_H4_n4 s_M1_n4 s_M2_n4 s_M3_n4 s_M4_n4 s_L1_n4 s_L2_n4 s_L3_n4 s_L4_n4 s_lows_n4 -v7.3

%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (4)  .... 441 -> 2282175
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load marginal_statistics_foster

tam_low = 4;     % 3, 4
tam_space = 20;  % 13, 20
tam_por_imagen = 441;
tam_total = tam_por_imagen*5175;

w = nw;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(nw(1,:))
    [nw_focus,ind_focus] = focus_on_center(nw(:,i),param(4).ind,pix_to_rem);
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(nw_focus,ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [1 i] 
    end
end
    s_H1_n4=s_H1s;    s_H2_n4=s_H2s;    s_H3_n4=s_H3s;    s_H4_n4=s_H4s;
    s_M1_n4=s_M1s;    s_M2_n4=s_M2s;    s_M3_n4=s_M3s;    s_M4_n4=s_M4s;
    s_L1_n4=s_L1s;    s_L2_n4=s_L2s;    s_L3_n4=s_L3s;    s_L4_n4=s_L4s;
    s_lows_n4=s_lows;
    
save samples_spatial_wav_tam_4_noise_full ...
s_H1_n4 s_H2_n4 s_H3_n4 s_H4_n4 s_M1_n4 s_M2_n4 s_M3_n4 s_M4_n4 s_L1_n4 s_L2_n4 s_L3_n4 s_L4_n4 s_lows_n4 -v7.3


%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  OK: NOW LETS COMPUTE THE TOTAL CORRELATION OF THE TRANSFORMED NOISE
%%%%%%%%%%%%%%%%%%%

clear all
load samples_spatial_tam_12_40_noise_low_dim
%load samples_spatial_tam_20_40_noise_low_dim
%load samples_spatial_tam_28_40_noise_low_dim
PARAMS.N_lay = 600;
num = round(0.85*129375);
    
for realiz=1:10

    ind_wavel = randperm(129375);
    indices_wavel = ind_wavel(1:num);

    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_nw_focus(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    Total_correl = [mean(Tn4')    std(Tn4')];
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];

    save total_correlation_spatial_noise_12_40_low_dim Total_correl Tn4

end

clear all
%load samples_spatial_tam_12_40_noise_low_dim
load samples_spatial_tam_20_40_noise_low_dim
%load samples_spatial_tam_28_40_noise_low_dim
PARAMS.N_lay = 600;
num = round(0.85*46575);
    
for realiz=1:10

    ind_wavel = randperm(46575);
    indices_wavel = ind_wavel(1:num);

    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_nw_focus(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    Total_correl = [mean(Tn4')    std(Tn4')];
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];

    save total_correlation_spatial_noise_20_40_low_dim Total_correl Tn4

end

clear all
%load samples_spatial_tam_12_40_noise_low_dim
%load samples_spatial_tam_20_40_noise_low_dim
load samples_spatial_tam_28_40_noise_low_dim
PARAMS.N_lay = 600;
num = round(0.9*5175);
    
for realiz=1:10

    ind_wavel = randperm(5175);
    indices_wavel = ind_wavel(1:num);

    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_nw_focus(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    Total_correl = [mean(Tn4')    std(Tn4')];
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];

    save total_correlation_spatial_noise_28_40_low_dim Total_correl Tn4

end

clear all
load samples_spatial_wav_tam_2_noise_full

PARAMS.N_lay = 600;
num = round(0.85*82800);
samples_n4 = [s_H1_n4;s_H2_n4;s_H3_n4;s_H4_n4;s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];
samples_n4_ml = [s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];

for realiz=1:10
    ind_wavel = randperm(82800);
    indices_wavel = ind_wavel(1:num);
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_n4_ml(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);
    Total_correl = [mean(Tn4')    std(Tn4')]
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];
    save total_correlation_noise_full_size_2b Total_correl Tn4 Tn4_converg
end

load samples_spatial_wav_tam_3_noise_full  
PARAMS.N_lay = 600;
num = round(0.85*46575);
samples_n4 = [s_H1_n4;s_H2_n4;s_H3_n4;s_H4_n4;s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];
samples_n4_ml = [s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];

for realiz=1:10
    ind_wavel = randperm(46575);
    indices_wavel = ind_wavel(1:num);
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_n4_ml(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);
    Total_correl = [mean(Tn4')    std(Tn4')]
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];
    save total_correlation_noise_full_size_3b Total_correl Tn4 Tn4_converg
end


load samples_spatial_wav_tam_4_noise_full    
PARAMS.N_lay = 600;
num = round(0.85*20700);
samples_n4 = [s_H1_n4;s_H2_n4;s_H3_n4;s_H4_n4;s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];
samples_n4_ml = [s_M1_n4;s_M2_n4;s_M3_n4;s_M4_n4;s_L1_n4;s_L2_n4;s_L3_n4;s_L4_n4;s_lows_n4];

for realiz=1:10
    ind_wavel = randperm(20700);
    indices_wavel = ind_wavel(1:num);
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_n4_ml(:,indices_wavel),PARAMS);
    Tn4(realiz) = PARAMSo.MI
    Tn4_conv(realiz,:) = integra_convergencia600(PARAMSo);
    Total_correl = [mean(Tn4')    std(Tn4')]
    Tn4_converg = [mean(Tn4_conv); std(Tn4_conv)];
    save total_correlation_noise_full_size_4b Total_correl Tn4 Tn4_converg
end
