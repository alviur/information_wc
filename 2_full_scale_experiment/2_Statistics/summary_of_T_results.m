
%%
%%
%%  EMPIRICAL RESULTS
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% WITH SUBSAMPLING (5d in space, 40d in wavelet)  
load total_correlation_spatial_12_40_low_dimens2b;T_3 = Total_correl;  % ALEX COMPARA ESTOS
load total_correlation_spatial_20_40_low_dim2b;T_5 = Total_correl;
load total_correlation_spatial_28_40_low_dim2b;T_7 = Total_correl;

load total_correlation_spatial_noise_12_40_low_dim;Tn_3 = Total_correl;
load total_correlation_spatial_noise_20_40_low_dim;Tn_5 = Total_correl;
load total_correlation_spatial_noise_28_40_low_dim;Tn_7 = Total_correl;

% Raw
[T_3 T_5 T_7],[Tn_3 Tn_5 Tn_7],

% Corrected
TC = [[T_3(1:4,:)/5;(T_3(5:7,1)-Tn_3(1))/40 T_3(5:7,2)/40] [T_5(1:4,:)/5;(T_5(5:7,1)-Tn_5(1))/40 T_5(5:7,2)/40] [T_7(1:4,:)/5;(T_7(5:7,1)-Tn_7(1))/40 T_7(5:7,2)/40]]
figure,errorbar(1:6,TC(1:6,1),TC(1:6,2),'o-','linewidth',2)
hold on,errorbar(1:6,TC(1:6,3),TC(1:6,4),'o-','linewidth',2)
hold on,errorbar(1:6,TC(1:6,5),TC(1:6,6),'o-','linewidth',2)
legend('\Delta x = 0.2 deg','\Delta x = 0.33 deg','\Delta x = 0.45 deg')
legend boxoff
ylabel({'Shared information per sensor';'Total Correlation (bits)'})
set(gcf,'color',[1 1 1])
set(gca,'XTick',[1 2 3 4 5 6],'XTickLabel',{'Luminance';'Lum post Von-Kries';'Brightness';'Nonlinear Contrast';'Linear V1';'Nonlinear V1'},'XTickLabelRotation',90)
axis([0.8 6.2 0 1.7])
title({'Redundancy reduction along the neural pathway';'(with subsampling)'})

%% NO SUBSAMPLING
load total_correlation_foster_full_size_2b;T_2=Total_correl;  % space = 100, wave = 81
load total_correlation_foster_full_size_3b;T_3=Total_correl;  % space = 169, wave = 181
load total_correlation_foster_full_size_4b;T_4=Total_correl;  % space = 400, wave = 321

load total_correlation_noise_full_size_2b;Tn_2=Total_correl;
load total_correlation_noise_full_size_3b;Tn_3=Total_correl;
load total_correlation_noise_full_size_4b;Tn_4=Total_correl;

[T_2 T_3 T_4],[Tn_2 Tn_3 Tn_4],


TC = [[T_2(1:4,:)/100;(T_2(5:7,1)-Tn_2(1))/81 T_2(5:7,2)/81] [T_3(1:4,:)/169;(T_3(5:7,1)-Tn_3(1))/181 T_3(5:7,2)/181] [T_4(1:4,:)/400;(T_4(5:7,1)-Tn_4(1))/321 T_4(5:7,2)/321]]
figure,errorbar(1:6,TC(1:6,1),TC(1:6,2),'o-','linewidth',2)
hold on,errorbar(1:6,TC(1:6,3),TC(1:6,4),'o-','linewidth',2)
hold on,errorbar(1:6,TC(1:6,5),TC(1:6,6),'o-','linewidth',2)
legend('\Delta x = 0.2 deg','\Delta x = 0.33 deg','\Delta x = 0.45 deg')
legend boxoff
ylabel({'Shared information per sensor';'Total Correlation (bits)'})
set(gcf,'color',[1 1 1])
set(gca,'XTick',[1 2 3 4 5 6],'XTickLabel',{'Luminance';'Lum post Von-Kries';'Brightness';'Nonlinear Contrast';'Linear V1';'Nonlinear V1'},'XTickLabelRotation',90)
axis([0.8 6.2 0 5])
title({'Redundancy reduction along the neural pathway';'(no subsampling)'})


%%
%%
%%  THEORETICAL RESULTS
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------------------------
% Interaction kernels for Neighborhoods with subsampling WW3 WW5 WW7
%----------------------------------------------------------------------------

clear all
load marginal_statistics_foster ind indices_w_focus
load parameters_WC_for_theoretical_TotCorr W W_focus_jesus ind_focus_jesus pix_to_rem_jesus alfa_m ind

% ind_wavelet = (1:length(alfa_m))';
% samples_ = scan_wavelet_for_spatial_samples(ind_wavelet,ind,4,0); 

ind_wavelet = (1:size(W_focus_jesus,2))';
indices_centrales3 = scan_wavelet_for_spatial_samples(ind_wavelet,ind_focus_jesus,3,1); 

po = 15;
ejemplo3 = [indices_centrales3(2,1).samples(:,po);indices_centrales3(2,2).samples(:,po);indices_centrales3(2,3).samples(:,po);indices_centrales3(2,4).samples(:,po);...
            indices_centrales3(3,1).samples(:,po);indices_centrales3(3,2).samples(:,po);indices_centrales3(3,3).samples(:,po);indices_centrales3(3,4).samples(:,po)];

% % This is to check that the example is well chosen
% coord = zeros(length(ejemplo3),4);
% for i=1:length(ejemplo3)
%     [coord(i,:),band] = coor_s_pyr(ejemplo3(i),ind_focus_jesus);
% end
% coord

indices_centrales5 = scan_wavelet_for_spatial_samples(ind_wavelet,ind_focus_jesus,5,1); 

po = 6;
ejemplo5 = [indices_centrales5(2,1).samples(:,po);indices_centrales5(2,2).samples(:,po);indices_centrales5(2,3).samples(:,po);indices_centrales5(2,4).samples(:,po);...
            indices_centrales5(3,1).samples(:,po);indices_centrales5(3,2).samples(:,po);indices_centrales5(3,3).samples(:,po);indices_centrales5(3,4).samples(:,po)];

% % This is to check that the example is well chosen
% ejemplo = ejemplo5;
% coord = zeros(length(ejemplo),4);
% for i=1:length(ejemplo)
%     [coord(i,:),band] = coor_s_pyr(ejemplo(i),ind_focus_jesus);
% end
% coord

indices_centrales7 = scan_wavelet_for_spatial_samples(ind_wavelet,ind_focus_jesus,7,1); 

po = 1;
ejemplo7 = [indices_centrales7(2,1).samples(:,po);indices_centrales7(2,2).samples(:,po);indices_centrales7(2,3).samples(:,po);indices_centrales7(2,4).samples(:,po);...
            indices_centrales7(3,1).samples(:,po);indices_centrales7(3,2).samples(:,po);indices_centrales7(3,3).samples(:,po);indices_centrales7(3,4).samples(:,po)];

%  % This is to check that the example is well chosen
%  ejemplo = ejemplo7;
%  coord = zeros(length(ejemplo),4);
%  for i=1:length(ejemplo)
%      [coord(i,:),band] = coor_s_pyr(ejemplo(i),ind_focus_jesus);
%  end
%  coord

WW3 = W_focus_jesus(ejemplo3,ejemplo3);
WW5 = W_focus_jesus(ejemplo5,ejemplo5);
WW7 = W_focus_jesus(ejemplo7,ejemplo7);

wavelet_under3 = 0*ind_wavelet;
wavelet_under3(ejemplo3) = 1;
wavelet_under5 = 0*ind_wavelet;
wavelet_under5(ejemplo5) = 1;
wavelet_under7 = 0*ind_wavelet;
wavelet_under7(ejemplo7) = 1;

figure(1),showSpyr(wavelet_under3,ind_focus_jesus)
figure(2),showSpyr(wavelet_under5,ind_focus_jesus)
figure(3),showSpyr(wavelet_under7,ind_focus_jesus)

%----------------------------------------------------------------------------
% Interaction kernels for Neighborhoods without subsampling 
%
%        WW_full_2 WW_full_3 WW_full_4         (really full)
%        WW_full_2ml WW_full_3ml WW_full_4ml   (only medium and low frequency)
%----------------------------------------------------------------------------

[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,2);
samples_space_scale_orient = [s_H1;s_H2;s_H3;s_H4;s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=10:10 % 1:16
    ejemplo_full2 = round(samples_space_scale_orient(:,i));
    wavelet_full2 = 0*ind_wavelet;
    wavelet_full2(ejemplo_full2) = 1;
    figure(100),showSpyr(wavelet_full2,ind_focus_jesus),title(num2str(i))
    pause
end
WW_full_2 = W_focus_jesus(ejemplo_full2,ejemplo_full2);
figure,imagesc(WW_full_2.^0.5)

[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,3);
samples_space_scale_orient = [s_H1;s_H2;s_H3;s_H4;s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=5:5 % 1:9
    try
    ejemplo_full3 = round(samples_space_scale_orient(:,i));
    wavelet_full3 = 0*ind_wavelet;
    wavelet_full3(ejemplo_full3) = 1;
    figure(101),showSpyr(wavelet_full3,ind_focus_jesus),title(num2str(i))

    pause
    end
end
WW_full_3 = W_focus_jesus(ejemplo_full3,ejemplo_full3);
figure,imagesc(WW_full_3.^0.5)

[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,4);
samples_space_scale_orient = [s_H1;s_H2;s_H3;s_H4;s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=1:4 % 1:4
    try
    ejemplo_full4 = round(samples_space_scale_orient(:,i));
    wavelet_full4 = 0*ind_wavelet;
    wavelet_full4(ejemplo_full4) = 1;
    figure(102),showSpyr(wavelet_full4,ind_focus_jesus),title(num2str(i))

    pause
    end
end
WW_full_4 = W_focus_jesus(ejemplo_full4,ejemplo_full4);
figure,imagesc(WW_full_4.^0.5)

%% It is fair to focus on the low frequencies: see the interactions!
%
% figure,plot(WW_full_4(1000:end,:)')
% interac = cumsum(WW_full_4(1000,:));
% figure,plot(interac/interac(end))
%


[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,2);
samples_space_scale_orient = [s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=10:10 % 1:16
    ejemplo_full2ml = round(samples_space_scale_orient(:,i));
    wavelet_full2ml = 0*ind_wavelet;
    wavelet_full2ml(ejemplo_full2ml) = 1;
    figure(100),showSpyr(wavelet_full2ml,ind_focus_jesus),title(num2str(i))
    pause
end
WW_full_2ml = W_focus_jesus(ejemplo_full2ml,ejemplo_full2ml);
figure,imagesc(WW_full_2ml.^0.5)

[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,3);
samples_space_scale_orient = [s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=5:5 % 1:9
    try
    ejemplo_full3ml = round(samples_space_scale_orient(:,i));
    wavelet_full3ml = 0*ind_wavelet;
    wavelet_full3ml(ejemplo_full3ml) = 1;
    figure(101),showSpyr(wavelet_full3ml,ind_focus_jesus),title(num2str(i))

    pause
    end
end
WW_full_3ml = W_focus_jesus(ejemplo_full3ml,ejemplo_full3ml);
figure,imagesc(WW_full_3ml.^0.5)

[s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(ind_wavelet,ind_focus_jesus,4);
samples_space_scale_orient = [s_M1;s_M2;s_M3;s_M4;s_L1;s_L2;s_L3;s_L4;s_low];
for i=1:4 % 1:4
    try
    ejemplo_full4ml = round(samples_space_scale_orient(:,i));
    wavelet_full4ml = 0*ind_wavelet;
    wavelet_full4ml(ejemplo_full4ml) = 1;
    figure(102),showSpyr(wavelet_full4ml,ind_focus_jesus),title(num2str(i))

    pause
    end
end
WW_full_4ml = W_focus_jesus(ejemplo_full4ml,ejemplo_full4ml);
figure,imagesc(WW_full_4ml.^0.5)


%----------------------------------------------------------------------------
% Auto-attenuation coefficients for Neighborhoods with subsampling 
%
%        alfa_sub_3, alfa_sub_5, alfa_sub_7         
%
% Auto-attenuation coefficients for Neighborhoods without subsampling 
%
%        alfa_2, alfa_3, alfa_4         
% 
%----------------------------------------------------------------------------

[alfa_m_focus, ind_focus] = focus_on_center(alfa_m,ind,pix_to_rem_jesus);

alfa_sub_3 = alfa_m_focus(ejemplo3);
alfa_sub_5 = alfa_m_focus(ejemplo5);
alfa_sub_7 = alfa_m_focus(ejemplo7);

alfa_2 = alfa_m_focus(ejemplo_full2ml);
alfa_3 = alfa_m_focus(ejemplo_full3ml);
alfa_4 = alfa_m_focus(ejemplo_full4ml);

%----------------------------------------------
%
%  And finally, the computation!  (Studeny formula with Wilson-Cowan jacobian)
%
%----------------------------------------------

%%  No subsampling case
%%

load samples_spatial_wav_tam_2_foster_full
samples_2_ml_y4 = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_2_ml_x4wc = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

% Sum of marginal entropies
sumH1 = 0;
sumH2 = 0;
for i=1:length(samples_2_ml_y4(:,1))
    sam = samples_2_ml_y4(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH1 = sumH1 + entropy_mm(p) + log2(delta);

    sam = samples_2_ml_x4wc(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH2 = sumH2 + entropy_mm(p) + log2(delta);
end

% Average of log of det of Jacobian
g = 0.5;    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOS CAMBIOS!
xm = mean(abs(samples_2_ml_x4wc)')';  %%%%%%%%%%%%%%%%%%%%%%%%%
epsilon = 1e-3*xm;                    %%%%%%%%%%%%%%%%%%%%%%%%%
average = 0;
N = length(sam);
Da = diag(alfa_2);
fact = 50;     fact=(1/((3*xm)^0.5))               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:N
    [f,dfdx] = saturation_f(samples_2_ml_x4wc(:,i),g,3*xm,1e-2*3*xm);
    J = g*inv(Da + fact*WW_full_2ml.*repmat(dfdx',[size(WW_full_2ml,1) 1])).*repmat(((Da*abs(samples_2_ml_x4wc(:,i)) + fact*WW_full_2ml*f).^(1-(1/g)))',[size(WW_full_2ml,1) 1] );
    % Since numbers are so small det is close to zero and log gives -inf ->
    % take eigenvalues and sum logs instead of log(product)
    lambda=abs(eig(J));
    average = average + sum(log2(lambda));
    if mod(i,500)==0
       [1 i N] 
    end
end
average = average/N
sumH1_2 = sumH1;
sumH2_2 = sumH2;
average_2 = average;
deltaT2 = sumH1_2 - sumH2_2 - average_2;

%%%%%%%%%%%%%%%%%%

load samples_spatial_wav_tam_3_foster_full
samples_2_ml_y4 = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_2_ml_x4wc = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

% Sum of marginal entropies
sumH1 = 0;
sumH2 = 0;
for i=1:length(samples_2_ml_y4(:,1))
    sam = samples_2_ml_y4(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH1 = sumH1 + entropy_mm(p) + log2(delta);

    sam = samples_2_ml_x4wc(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH2 = sumH2 + entropy_mm(p) + log2(delta);
end

% Average of log of det of Jacobian    %%%% LOS CAMBIOS
g = 0.6;
xm = mean(abs(samples_2_ml_x4wc)')';
epsilon = 1e-3*xm;
average = 0;
N = length(sam);
Da = diag(alfa_3);

for i=1:N
    [f,dfdx] = saturation_f(samples_2_ml_x4wc(:,i),g,3*xm,1e-2*3*xm);
    J = g*inv(Da + WW_full_3ml.*repmat(dfdx',[size(WW_full_3ml,1) 1])).*repmat(((Da*abs(samples_2_ml_x4wc(:,i)) + WW_full_3ml*f).^(1-g))',[size(WW_full_3ml,1) 1] );
    % Since numbers are so small det is close to zero and log gives -inf ->
    % take eigenvalues and sum logs instead of log(product)
    lambda=abs(eig(J));
    average = average + sum(log2(abs(lambda)));
    if mod(i,500)==0
       [2 i N]
    end
end
average = average/N

sumH1_3 = sumH1;
sumH2_3 = sumH2;
average_3 = average;
deltaT3 = sumH1_3 - sumH2_3 + average_3;

%%%%%%%%%%%%%%%%%%

load samples_spatial_wav_tam_4_foster_full
samples_2_ml_y4 = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_2_ml_x4wc = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

% Sum of marginal entropies
sumH1 = 0;
sumH2 = 0;
for i=1:length(samples_2_ml_y4(:,1))
    sam = samples_2_ml_y4(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH1 = sumH1 + entropy_mm(p) + log2(delta);

    sam = samples_2_ml_x4wc(i,:);
    [p,domin] = hist(sam,round(sqrt(length(sam))));
    delta = domin(2)-domin(1);
    p = p/(sum(p)*delta);
    sumH2 = sumH2 + entropy_mm(p) + log2(delta);
end

% Average of log of det of Jacobian   %%%% LOS CAMBIOS
g = 0.6;
xm = mean(abs(samples_2_ml_x4wc)')';
epsilon = 1e-3*xm;
average = 0;
N = length(sam);
Da = diag(alfa_4);

for i=1:N
    [f,dfdx] = saturation_f(samples_2_ml_x4wc(:,i),g,3*xm,1e-2*3*xm);
    J = g*inv(Da + WW_full_4ml.*repmat(dfdx',[size(WW_full_4ml,1) 1])).*repmat(((Da*abs(samples_2_ml_x4wc(:,i)) + WW_full_4ml*f).^(1-g))',[size(WW_full_4ml,1) 1] );
    % Since numbers are so small det is close to zero and log gives -inf ->
    % take eigenvalues and sum logs instead of log(product)
    lambda=abs(eig(J));
    average = average + sum(log2(lambda));
    if mod(i,500)==0
       [3 i N]
    end
end
average = average/N

sumH1_4 = sumH1;
sumH2_4 = sumH2;
average_4 = average;
deltaT4 = sumH1_4 - sumH2_4 + average_4;

%
[deltaT2/81 deltaT3/181 deltaT4/321]