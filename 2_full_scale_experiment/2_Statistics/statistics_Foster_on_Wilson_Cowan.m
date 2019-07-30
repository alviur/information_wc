%%
%%  STATISTICS OF FOSTER DATA IN A PSYCHOPHYSICALLY TUNED WILSON-COWAN CASCADE
%%
%%    In this script I gather the data and compute the relevant statistics
%%
%%      Input data: 
%%         * Original images and divisive normalization cascade:
%%              /media/disk/vista/Papers/2017_Information_Flow/DataFoster/processed_40x40/samples_general_40x40_X.mat
%%                                                                                        samples_general_40x40_X_y1.mat
%%         * Results of the Wilson-Cowan integration:
%%              /media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/Results_integration_TF/X.mat
%%
%%      Output (corrupted data in batch 10 were removed):
%%
%%         * All raw data in a single file:
%%              /media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/Results_integration_TF/all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc 
%%           which contains 5175 images at the different layers:
%%                y1_original_ok 
%%                y1_ok 
%%                x1_ok 
%%                x3_ok 
%%                y4_ok 
%%                x4_ok 
%%                x4wc_ok
%%
%%         * Marginal statistics
%%               marginal_statistics_foster
%%      
%%         * Pairs of mutual information between coefficients
%%                 mutuals_luminance_foster  Iyo1
%% 
%%                 mutuals_luminance_after_von_kries_foster Iy1
%% 
%%                 mutuals_brightness_foster  Ix1
%% 
%%                 mutuals_contrast_foster  Ix3
%% 
%%                 mutuals_y4_foster
%%                      Iy4_a2 Iy4_b2 Iy4_c2
%%                 mutuals_x4dn_foster
%%                      Ix4dn_a2 Ix4dn_b2 Ix4dn_c2
%%                 mutuals_x4wc_foster
%%                      Ix4wc_a2 Ix4wc_b2 Ix4wc_c2
%% 
%%         * [spatially subsampled] image patches gathered to compute total correlation
%%           (samples covering certain spatial field -either in the spatial domain or in the wavelet domain-)
%%
%%                 samples_spatial_tam_12_40_foster (5 points per spatial location -corners and center-, only two scales)
%%                        samples_space_y1o 
%%                        samples_space_y1 
%%                        samples_space_x1 
%%                        samples_space_x3 
%%                        samples_y4_focus 
%%                        samples_x4wc_focus 
%%                        samples_x4dn_focus
%%
%%                samples_spatial_tam_20_40_foster
%%
%%                samples_spatial_tam_28_40_foster
%%
%%                samples_spatial_tam_12_40_foster_high_dim  (9 points per spatial location, only two scales)
%%                samples_spatial_tam_20_40_foster_high_dim  
%%                samples_spatial_tam_28_40_foster_high_dim  
%%                
%%                samples_spatial_tam_12_40_foster_even_higher_dim  (9 points per spatial location, all 3 scales)
%%                samples_spatial_tam_20_40_foster_even_higher_dim  
%%                samples_spatial_tam_28_40_foster_even_higher_dim  
%%                
%%       * Total correlation on spatially subsampled images
%%
%%                total_correlation_spatial_12_40_low_dimens
%%                total_correlation_spatial_20_40_low_dimens
%%                total_correlation_spatial_28_40_low_dimens
%%
%%                total_correlation_spatial_12_40_high_dimens
%%                total_correlation_spatial_20_40_high_dimens
%%                total_correlation_spatial_28_40_high_dimens
%%
%%                total_correlation_spatial_12_40_even_higher_dimens
%%                total_correlation_spatial_20_40_even_higher_dimens
%%                total_correlation_spatial_28_40_even_higher_dimens
%%
%%         * [Full] image patches gathered to compute total correlation
%%           (samples covering certain spatial field -either in the spatial domain or in the wavelet domain-)
%%
%%                samples_spatial_wav_tam_2_foster_full 
%%                         samples_space_y1o_full 
%%                         samples_space_y1_full 
%%                         samples_space_x1_full 
%%                         samples_space_x3_full 
%%                         s_H1_y4 s_H2_y4 s_H3_y4 s_H4_y4 
%%                         s_M1_y4 s_M2_y4 s_M3_y4 s_M4_y4 
%%                         s_L1_y4 s_L2_y4 s_L3_y4 s_L4_y4 s_lows_y4
%%                         s_H1_x4wc s_H2_x4wc s_H3_x4wc s_H4_x4wc 
%%                         s_M1_x4wc s_M2_x4wc s_M3_x4wc s_M4_x4wc 
%%                         s_L1_x4wc s_L2_x4wc s_L3_x4wc s_L4_x4wc s_lows_x4wc
%%                         s_H1_x4dn s_H2_x4dn s_H3_x4dn s_H4_x4dn 
%%                         s_M1_x4dn s_M2_x4dn s_M3_x4dn s_M4_x4dn 
%%                         s_L1_x4dn s_L2_x4dn s_L3_x4dn s_L4_x4dn s_lows_x4dn 
%%
%%         * Total correlation results (600 iterations to ensure convergence!)
%%
%%               total_correlation_foster_full_size_2b (600 iterations)
%%               total_correlation_foster_full_size_3b 
%%               total_correlation_foster_full_size_4b 
%%
%%               total_correlation_foster_full_size_2 (only 150 iterat!) 
%%               total_correlation_foster_full_size_3 
%%               total_correlation_foster_full_size_4 
%%


% Luminace images and responses (from y1 to x4)

%% 1.Load data batch-wise and put it in a single vector
timeFiles = '/media/disk/databases/BBDD_video_image/Image_Statistic/DataFoster/processed_40x40/';% Valencia

name = 'samples_general_40x40_';

y1_original = [];
y1 = [];
x1 = [];
x3 = [];
y4 = [];
x4 = [];
x4wc = [];

inicio = 0;
for batch=1:20 % Loop over batches
    i
    if(batch~=3)
        % Load image batch here
        load([timeFiles,name,num2str(batch),'.mat'])
        %load([timeFiles,nameJ,num2str(batch),'.mat'])
        display(['Batch ',num2str(batch),' loaded'])    
        numImages = size(samplesA4x,2);

        x1 = [x1 samplesA1x/260^1.25];   % See notebook (20 jun 2019) for the mistake in normalization of brightness
        x3 = [x3 samplesA3x];
        x4 = [x4 samplesA4x];
        y4 = [y4 samplesA4y];

        % Load image batch here
        load([timeFiles,name,num2str(batch),'_y1.mat'])

        y1_original = [y1_original samplesA1y_sin_v];
        y1 = [y1 samplesA1y];
        
        load(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/Resuts_integration_TF_2/',num2str(batch)])
        x4wc = [x4wc integrated];
        
        cuals(batch).indices = inicio+1:inicio+length(samplesA1y(1,:));
        inicio = inicio+length(samplesA1y(1,:));
    else
        cuals(batch).indices = [];
    end
end

ind =[40    40;
    40    40;
    40    40;
    40    40;
    40    40;
    20    20;
    20    20;
    20    20;
    20    20;
    10    10;
    10    10;
    10    10;
    10    10;
     5     5];
 
%%%% MAL: 2415 - 2484 

y1_original_ok = y1_original(:,[1:2415 2485:5244]);
y1_ok = y1(:,[1:2415 2485:5244]);
x1_ok = x1(:,[1:2415 2485:5244]);
x3_ok = x3(:,[1:2415 2485:5244]);
y4_ok = y4(:,[1:2415 2485:5244]);
x4_ok = x4(:,[1:2415 2485:5244]);
x4wc_ok = sign(x4_ok).*x4wc(:,[1:2415 2485:5244]);


for batch = 4:20
indi = cuals(batch).indices;
for i = indi(1):10:indi(end)
    
%     L1 = mean(y1(:,i));
%     A1 = std(y1(:,i))*sqrt(2);
%     c = A1/L1;  
% 
%     C(i) = c;
%     L(i) = L1;
% 
%     im = reshape(y1(:,i),[40 40]);
%     imc = im(9:32,9:32);
% 
%     % L1 = mean(imc(:));
%     A1 = std(imc(:))*sqrt(2);
%     c = A1/L1;  
%     
%     Cc(i) = c;
%     Lc(i) = L1;

    figure(1),subplot(151),colormap gray,imagesc(reshape(y1_original(:,i),[40 40]),[0 250]),title(['image ',num2str(i)])
              subplot(152),colormap gray,imagesc(reshape(y1(:,i),[40 40]),[0 250]),title(['image ',num2str(i)])
              subplot(153),colormap gray,imagesc(reshape(x1(:,i),[40 40]),[0 150])
              subplot(154),colormap gray,plot(reshape(x1(:,i),[40 40]))
              subplot(155),colormap gray,imagesc(reshape(x3(:,i),[40 40]),[-3 3])
              figure(2),showSpyr(y4(:,i),ind),title('y')
              figure(3),showSpyr(x4(:,i),ind),title('x4')
              figure(4),showSpyr(sign(x4(:,i)).*x4wc(:,i),ind),title('x4 WC'),title([num2str(batch),'   ',num2str(i)])
              pause(0.5)
end
end

cd /media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/Resuts_integration_TF_2
save all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc y1_original_ok y1_ok x1_ok x3_ok y4_ok x4_ok x4wc_ok

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%% Marginal statistics computed from stored samples
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% central part 
indices_spatial = reshape(1:1600,[40 40]);
borde = 8;
indices_central_space = indices_spatial(borde+1:end-borde,borde+1:end-borde);

%%
%%  LUMINANCE BEFORE VON-KRIES
%%

Lso = [linspace(0,3600,5*round(sqrt(5200*(40-2*borde).^2)))];
lumin_datao = y1_original_ok(indices_central_space,:);
po = hist(lumin_datao(:),Lso);
po = po/(sum(po)*(Lso(2)-Lso(1)));
norm = sum(po*(Lso(2)-Lso(1)))
figure(1),subplot(121),loglog(Lso,po,'linewidth',1.5),xlabel('Luminance (cd/m2)'),
axis([5e-1 400 1e-6 1e-1])
figure(1),subplot(122),plot(Lso,po,'linewidth',1.5),xlabel('Luminance (cd/m2)'),
axis([5e-1 300 0 1e-1])

%%
%%  LUMINANCE
%%

Ls = [linspace(0,3600,5*round(sqrt(5200*(40-2*borde).^2)))];
lumin_data = y1_ok(indices_central_space,:);
p = hist(lumin_data(:),Ls);
p = p/(sum(p)*(Ls(2)-Ls(1)));
norm = sum(p*(Ls(2)-Ls(1)))
figure(1),subplot(121),hold on,loglog(Ls,p,'linewidth',1.5),xlabel('Luminance (cd/m2)'),
axis([5e-1 400 1e-6 1e-1])
figure(1),subplot(122),hold on,plot(Ls,p,'linewidth',1.5),xlabel('Luminance (cd/m2)'),
axis([5e-1 300 0 1e-1])

%%
%%  BRIGHTNESS
%%

%% 45/23 is the factor to re-scale the brightness to put in the same range than luminace (both with the 90% percentile)
%%   extra factor (400/240) es por la correccion cutre de la brightness

Bs = [linspace(0,3600,5*round(sqrt(5200*(40-2*borde).^2)))];
bright_data = ((400/240)*45/23)*x1_ok(indices_central_space,:);
pb = hist(bright_data(:),Bs);
pb = pb/(sum(pb)*(Bs(2)-Bs(1)));
norm = sum(pb*(Bs(2)-Bs(1)))
figure(1),subplot(121),hold on,loglog(Bs,pb,'linewidth',1.5),xlabel({'Luminance (cd/m2)';'Brightness (linearly rescaled units)'}),ylabel('probability')
axis([4e-1 1000 0.8e-6 1])
legend('Luminance','Lumin. post Von-Kries','Brightness')
legend(gca,'boxoff')
title({'Luminance and brightness statistics';'(spatial domain)'})
set(gcf,'color',[1 1 1])
figure(1),subplot(122),hold on,plot(Bs,pb,'linewidth',1.5),xlabel({'Luminance (cd/m2)';'Brightness (linearly rescaled units)'})
axis([3e-1 200 1e-6 2e-1])
legend('Luminance','Lumin. post Von-Kries','Brightness')
legend(gca,'boxoff')
set(gcf,'color',[1 1 1])

%%
%%  CONTRAST
%%

C = [linspace(-2,2,0.5*round(sqrt(5200*(40-2*borde).^2)))];
contrast_data = x3_ok(indices_central_space,:);
pc = hist(contrast_data(:),C);
pc = pc/(sum(pc)*(C(2)-C(1)));
norm = sum(pc*(C(2)-C(1)))
figure(3),semilogy(C,pc,'linewidth',1.5),xlabel('Contrast'),ylabel('probability')
axis([-1.5 1.5 1e-2 10])
set(gcf,'color',[1 1 1])
title({'Statistics of nonlinear contrast';'(spatial domain)'})

%figure(6),plot(C,pc,'linewidth',1.5),xlabel('Brightness'),ylabel('p(x^1)')
%axis([-1.5 1.5 1e-2 1])

%%
%% RESPONSES IN THE LINEAR V1 DOMAIN
%%

% cd('C:\disco_portable\mundo_irreal\jesus\model_WC_flow\Results_integration_TF_IPLcolor')
% load samples_40x40_x4wc
h = genpath('/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/jesus');
addpath(h)


% Dimension of the wavelet
ind = [40 40;40 40;40 40;40 40;40 40;20 20;20 20;20 20;20 20;10 10;10 10;10 10;10 10;5 5];
d = sum(prod(ind,2));
indices_w = (1:d)';

% Central part of the image

% [indices_w_focus, ind_focus] = focus_on_center(indices_w,ind,[10 10 10 10 10 5 5 5 5 3 3 3 3 2]);
[indices_w_focus, ind_focus] = focus_on_center(indices_w,ind,[4 4 4 4 4 2 2 2 2 1 1 1 1 1]);

y4_focus = y4_ok(indices_w_focus,:);
% x4dn_focus = x4dn(indices_w_focus,:);
x4wc_focus = x4wc_ok(indices_w_focus,:);

indices_high = pyrBandIndices(ind_focus,1);
indices_scale_orient(1,1).indi = pyrBandIndices(ind_focus,2);
indices_scale_orient(1,2).indi = pyrBandIndices(ind_focus,3);
indices_scale_orient(1,3).indi = pyrBandIndices(ind_focus,4);
indices_scale_orient(1,4).indi = pyrBandIndices(ind_focus,5);
indices_scale_orient(2,1).indi = pyrBandIndices(ind_focus,6);
indices_scale_orient(2,2).indi = pyrBandIndices(ind_focus,7);
indices_scale_orient(2,3).indi = pyrBandIndices(ind_focus,8);
indices_scale_orient(2,4).indi = pyrBandIndices(ind_focus,9);
indices_scale_orient(3,1).indi = pyrBandIndices(ind_focus,10);
indices_scale_orient(3,2).indi = pyrBandIndices(ind_focus,11);
indices_scale_orient(3,3).indi = pyrBandIndices(ind_focus,12);
indices_scale_orient(3,4).indi = pyrBandIndices(ind_focus,13);
indices_low = pyrBandIndices(ind_focus,14);

ind_scale1 = [indices_scale_orient(1,1).indi';
              indices_scale_orient(1,2).indi';
              indices_scale_orient(1,3).indi';
              indices_scale_orient(1,4).indi'];
ind_scale2 = [indices_scale_orient(2,1).indi';
              indices_scale_orient(2,2).indi';
              indices_scale_orient(2,3).indi';
              indices_scale_orient(2,4).indi'];
ind_scale3 = [indices_scale_orient(3,1).indi';
              indices_scale_orient(3,2).indi';
              indices_scale_orient(3,3).indi';
              indices_scale_orient(3,4).indi'];  

%%%%%%%%%%%% 

w1 = [linspace(-1,1,round(sqrt(5000*prod(ind_focus(1,:)))))];
data = y4_focus(ind_scale1,:);
p1 = hist(data(:),w1);
p1 = p1/(sum(p1)*(w1(2)-w1(1)));
norm = sum(p1*(w1(2)-w1(1)))

w2 = [linspace(-5.5,5.5,round(sqrt(5000*prod(ind_focus(6,:)))))];
data = y4_focus(ind_scale2,:);
p2 = hist(data(:),w2);
p2 = p2/(sum(p2)*(w2(2)-w2(1)));
norm = sum(p2*(w2(2)-w2(1)))

w3 = [linspace(-21,21,round(sqrt(5000*prod(ind_focus(10,:)))))];
data = y4_focus(ind_scale3,:);
p3 = hist(data(:),w3);
p3 = p3/(sum(p3)*(w3(2)-w3(1)));
norm = sum(p3*(w3(2)-w3(1)))

w4 = [linspace(-90,90,round(sqrt(5000*prod(ind_focus(14,:)))))];
data = y4_focus(indices_low,:);
p4 = hist(data(:),w4);
p4 = p4/(sum(p4)*(w4(2)-w4(1)));
norm = sum(p4*(w4(2)-w4(1)))

figure,semilogy(w1,p1,'linewidth',1.5),hold on
       semilogy(w2,p2,'linewidth',1.5)
       semilogy(w3,p3,'linewidth',1.5)
       xlabel('Linear V1 response')
       ylabel('probability')
       title({'Statistics of Linear V1 responses';'(Wavelet-like domain)'})
       legend('High freq. (15 cpd)','Med. freq. (7.5 cpd)','Low freq. (3.2 cpd)')
       legend(gca,'boxoff')
set(gcf,'color',[1 1 1])
axis([-15 15 1e-4 1e2])

%%
%% RESPONSES IN THE NONLINEAR WILSON-COWAN V1 DOMAIN
%%

wc1 = [linspace(-0.16e-3,0.16e-3,round(sqrt(5000*prod(ind_focus(1,:)))))];
data = x4wc_focus(ind_scale1,:);
pwc1 = hist(data(:),wc1);
pwc1 = pwc1/(sum(pwc1)*(wc1(2)-wc1(1)));
norm = sum(pwc1*(wc1(2)-wc1(1)))

wc2 = [linspace(-0.6e-3,0.6e-3,round(sqrt(5000*prod(ind_focus(6,:)))))];
data = x4wc_focus(ind_scale2,:);
pwc2 = hist(data(:),wc2);
pwc2 = pwc2/(sum(pwc2)*(wc2(2)-wc2(1)));
norm = sum(pwc2*(wc2(2)-wc2(1)))

wc3 = [linspace(-1e-3,1e-3,round(sqrt(5000*prod(ind_focus(10,:)))))];
data = x4wc_focus(ind_scale3,:);
pwc3 = hist(data(:),wc3);
pwc3 = pwc3/(sum(pwc3)*(wc3(2)-wc3(1)));
norm = sum(pwc3*(wc3(2)-wc3(1)))

wc4 = [linspace(-0.003,0.003,round(sqrt(5000*prod(ind_focus(14,:)))))];
data = x4wc_focus(indices_low,:);
pwc4 = hist(data(:),wc4);
pwc4 = pwc4/(sum(pwc4)*(wc4(2)-wc4(1)));
norm = sum(pwc4*(wc4(2)-wc4(1)))

figure,semilogy(wc1,pwc1,'linewidth',1.5),hold on
       semilogy(wc2,pwc2,'linewidth',1.5)
       semilogy(wc3,pwc3,'linewidth',1.5)
       xlabel('Nonlinear Wilson-Cowan V1 response')
       ylabel('probability')
       title({'Statistics of Wilson-Cowan V1 responses';'(Wavelet-like domain)'})
       legend('High freq. (15 cpd)','Med. freq. (7.5 cpd)','Low freq. (3.2 cpd)')
       legend(gca,'boxoff')
set(gcf,'color',[1 1 1])
axis([-8e-4 8e-4 1e0 1e5])

clear y1_original_ok y1_ok x1_ok x3_ok y4_ok x4_ok x4wc_ok
save marginal_statistics_foster


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%%  PAIRS OF MUTUAL INFORMATION
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Luminance before
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
y1 = y1_original_ok;
for realiz = 1:10
    indi = randperm(5175);
    num = 0.8*5175;
    for i=1:22
        [0 realiz i]
        for j=1:22
            rv1 = [y1(indices_central_space(i,j),indi(1:num))   y1(indices_central_space(i+1,j),indi(1:num))   y1(indices_central_space(i,j+1),indi(1:num))   y1(indices_central_space(i+1,j+1),indi(1:num))];
            rv2 = [y1(indices_central_space(11,11),indi(1:num)) y1(indices_central_space(11+1,11),indi(1:num)) y1(indices_central_space(11,11+1),indi(1:num)) y1(indices_central_space(11+1,11+1),indi(1:num))];
            Iy1(i,j,realiz) = mutual_information_4(rv1,rv2,4*round(sqrt(sqrt(length(rv1)))));
        end
    end
end
Iyo1 = Iy1;
save mutuals_luminance_foster Iyo1

%% Luminance after
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
y1 = y1_ok;
for realiz = 1:10
    indi = randperm(5175);
    num = 0.8*5175;
    for i=1:22
        [1 realiz i]
        for j=1:22
            rv1 = [y1(indices_central_space(i,j),indi(1:num))   y1(indices_central_space(i+1,j),indi(1:num))   y1(indices_central_space(i,j+1),indi(1:num))   y1(indices_central_space(i+1,j+1),indi(1:num))];
            rv2 = [y1(indices_central_space(11,11),indi(1:num)) y1(indices_central_space(11+1,11),indi(1:num)) y1(indices_central_space(11,11+1),indi(1:num)) y1(indices_central_space(11+1,11+1),indi(1:num))];
            Iy1(i,j,realiz) = mutual_information_4(rv1,rv2,4*round(sqrt(sqrt(length(rv1)))));
        end
    end
end
save mutuals_luminance_after_von_kries_foster Iy1

1+1

%%% Brightness
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
x1 = x1_ok;
for realiz = 1:10
    indi = randperm(5175);
    num = 0.8*5175;   
    for i=1:22
        [2 realiz i]
        for j=1:22
            rv1 = [x1(indices_central_space(i,j),indi(1:num))   x1(indices_central_space(i+1,j),indi(1:num))   x1(indices_central_space(i,j+1),indi(1:num))   x1(indices_central_space(i+1,j+1),indi(1:num))];
            rv2 = [x1(indices_central_space(11,11),indi(1:num)) x1(indices_central_space(11+1,11),indi(1:num)) x1(indices_central_space(11,11+1),indi(1:num)) x1(indices_central_space(11+1,11+1),indi(1:num))];
            Ix1(i,j,realiz) = mutual_information_4(rv1,rv2,4*round(sqrt(sqrt(length(rv1)))));
        end
    end
end
save mutuals_brightness_foster Ix1

%%% Contrast
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
x3 = x3_ok;
for realiz = 1:10
    indi = randperm(5175);
    num = 0.8*5175;   
    for i=1:22
        [3 realiz i]
        for j=1:22
            rv1 = [x3(indices_central_space(i,j),indi(1:num))   x3(indices_central_space(i+1,j),indi(1:num))   x3(indices_central_space(i,j+1),indi(1:num))   x3(indices_central_space(i+1,j+1),indi(1:num))];
            rv2 = [x3(indices_central_space(11,11),indi(1:num)) x3(indices_central_space(11+1,11),indi(1:num)) x3(indices_central_space(11,11+1),indi(1:num)) x3(indices_central_space(11+1,11+1),indi(1:num))];
            Ix3(i,j,realiz) = mutual_information_4(rv1,rv2,4*round(sqrt(sqrt(length(rv1)))));
        end
    end
end
save mutuals_contrast_foster Ix3

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%

load mutuals_luminance_foster
load mutuals_luminance_after_von_kries_foster
load mutuals_brightness_foster
load mutuals_contrast_foster

[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(22,22,1,40,40,1);
x = x - max(x(:))/2;
y = y - max(y(:))/2;

IIyo1 = mean(Iyo1,3);
for i=1:22
    for j=1:22
        sIyo1(i,j) = std(Iyo1(i,j,:));
    end
end
delat_Iyo1 = mean(sIyo1(:));

IIy1 = mean(Iy1,3);
for i=1:22
    for j=1:22
        sIy1(i,j) = std(Iy1(i,j,:));
    end
end
delat_Iy1 = mean(sIy1(:));

IIx1 = mean(Ix1,3);
for i=1:22
    for j=1:22
        sIx1(i,j) = std(Ix1(i,j,:));
    end
end
delat_Ix1 = mean(sIx1(:));

IIx3 = mean(Ix3,3);
for i=1:22
    for j=1:22
        sIx3(i,j) = std(Ix3(i,j,:));
    end
end
delat_Ix3 = mean(sIx3(:));

IIyo1(11,11) = 0;
IIy1(11,11) = 0;
IIx1(11,11) = 0;
IIx3(11,11) = 0;
figure,subplot(141),mesh(x(1,:),y(:,1),IIyo1),axis([-0.3 0.3 -0.3 0.3 0 2.25]),title({'Mutual Inform. (in bits)';'Luminance in the spatial domain'})
xlabel('\Delta x (deg)'),ylabel('\Delta y (deg)')
subplot(142),mesh(x(1,:),y(:,1),IIy1),axis([-0.3 0.3 -0.3 0.3 0 2.25]),title({'Mutual Inform. (in bits)';'Luminance after Von-Kries'})
subplot(143),mesh(x(1,:),y(:,1),IIx1),axis([-0.3 0.3 -0.3 0.3 0 2.25]),title({'Mutual Inform. (in bits)';'Brightness'})
subplot(144),mesh(x(1,:),y(:,1),IIx3),axis([-0.3 0.3 -0.3 0.3 0 2.25]),title({'Mutual Inform. (in bits)';'Contrast'})
[delat_Iyo1 delat_Iy1 delat_Ix1 delat_Ix3]
set(gcf,'color',[1 1 1])

%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%

%
% LINEAR V1
%
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
y4_focus = y4_ok(indices_w_focus,:);
factor_hist = 1;nonflat = 1:5175;
[position,band] = pos_s_pyr([2 1 8 8],ind_focus);
for realiz = 1:10,[1 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = y4_focus(position,nonflat(indi(1:num)));
    for i = 1:length(y4_focus(:,1))
        %[1 realiz i]
        rv2 = y4_focus(i,nonflat(indi(1:num)));
        Iy4_a2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([3 1 4 4],ind_focus);
for realiz = 1:10,[2 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = y4_focus(position,nonflat(indi(1:num)));
    for i = 1:length(y4_focus(:,1))
        %[2 realiz i]
        rv2 = y4_focus(i,nonflat(indi(1:num)));
        Iy4_b2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([2 2 8 8],ind_focus);
for realiz = 1:10,[3 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = y4_focus(position,nonflat(indi(1:num)));
    for i = 1:length(y4_focus(:,1))
        %[3 realiz i]
        rv2 = y4_focus(i,nonflat(indi(1:num)));
        Iy4_c2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
save mutuals_y4_foster Iy4_a2 Iy4_b2 Iy4_c2

%
% NON-LINEAR V1 WC
%
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
x4wc_focus = x4wc_ok(indices_w_focus,:);
factor_hist = 1;nonflat = 1:5175; [position,band] = pos_s_pyr([2 1 8 8],ind_focus);
for realiz = 1:10,[1 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4wc_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4wc_focus(:,1))
        %[1 realiz i]
        rv2 = x4wc_focus(i,nonflat(indi(1:num)));
        Ix4wc_a2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([3 1 4 4],ind_focus);
for realiz = 1:10,[2 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4wc_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4wc_focus(:,1))
        %[2 realiz i]
        rv2 = x4wc_focus(i,nonflat(indi(1:num)));
        Ix4wc_b2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([2 2 8 8],ind_focus);
for realiz = 1:10,[3 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4wc_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4wc_focus(:,1))
        %[3 realiz i]
        rv2 = x4wc_focus(i,nonflat(indi(1:num)));
        Ix4wc_c2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
save mutuals_x4wc_foster Ix4wc_a2 Ix4wc_b2 Ix4wc_c2


%
% NON-LINEAR V1 DN
%

load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster
x4dn_focus = x4_ok(indices_w_focus,:);
factor_hist = 1;nonflat = 1:5175; [position,band] = pos_s_pyr([2 1 8 8],ind_focus);
for realiz = 1:10,[1 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4dn_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4dn_focus(:,1))
        %[1 realiz i]
        rv2 = x4dn_focus(i,nonflat(indi(1:num)));
        Ix4dn_a2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([3 1 4 4],ind_focus);
for realiz = 1:10,[2 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4dn_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4dn_focus(:,1))
        %[2 realiz i]
        rv2 = x4dn_focus(i,nonflat(indi(1:num)));
        Ix4dn_b2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
[position,band] = pos_s_pyr([2 2 8 8],ind_focus);
for realiz = 1:10,[3 realiz]
    indi = randperm(length(nonflat));
    num = round(0.8*length(nonflat));    
    rv1 = x4dn_focus(position,nonflat(indi(1:num)));
    for i = 1:length(x4dn_focus(:,1))
        %[3 realiz i]
        rv2 = x4dn_focus(i,nonflat(indi(1:num)));
        Ix4dn_c2(i,realiz) = mutual_information_4(rv1,rv2,round(factor_hist*sqrt(sqrt(length(rv1)))));
    end
end
save mutuals_x4dn_foster Ix4dn_a2 Ix4dn_b2 Ix4dn_c2

load mutuals_x4wc_foster 
Ix4wc_a=Ix4wc_a2; 
Ix4wc_b=Ix4wc_b2; 
Ix4wc_c=Ix4wc_c2;
load mutuals_x4dn_foster 
Ix4wc_a=Ix4dn_a2; 
Ix4wc_b=Ix4dn_b2; 
Ix4wc_c=Ix4dn_c2;
load mutuals_y4_foster 
Iy4_a=Iy4_a2; 
Iy4_b=Iy4_b2; 
Iy4_c=Iy4_c2;

[p_a,band] = pos_s_pyr([2 1 8 8],ind_focus);
[p_b,band] = pos_s_pyr([3 1 4 4],ind_focus);
[p_c,band] = pos_s_pyr([2 2 8 8],ind_focus);
Iy4_a(p_a,:) = 0;
Iy4_b(p_b,:) = 0;
Iy4_c(p_c,:) = 0;
Ma = max(Iy4_a(:))
Mb = max(Iy4_b(:))
Mc = max(Iy4_c(:))
Ix4wc_a(p_a,:) = 0;
Ix4wc_b(p_b,:) = 0;
Ix4wc_c(p_c,:) = 0;
Maw = max(Ix4wc_a(:))
Mbw = max(Ix4wc_b(:))
Mcw = max(Ix4wc_c(:))
M = max([Ma Mb Mc Maw Mbw Mcw])
expo= 0.9;
figure,showSpyr((mean(Iy4_a,2)).^expo,ind_focus,[0 M].^expo,1,1),title('y 2nd scale vertical')
figure,showSpyr((mean(Iy4_b,2)).^expo,ind_focus,[0 M].^expo,1,1),title('y 3rd scale vertical')
figure,showSpyr((mean(Iy4_c,2)).^expo,ind_focus,[0 M].^expo,1,1),title('y 2nd scale diagonal')
expo= 0.9;
figure,showSpyr((mean(Ix4wc_a,2)).^expo,ind_focus,[0 M].^expo,1,1),title('x 2nd scale vertical')
figure,showSpyr((mean(Ix4wc_b,2)).^expo,ind_focus,[0 M].^expo,1,1),title('x 3rd scale vertical')
figure,showSpyr((mean(Ix4wc_c,2)).^expo,ind_focus,[0 M].^expo,1,1),title('x 2nd scale diagonal')

%%%%% Lets compute some summary numbers

load mutuals_luminance_foster
load mutuals_luminance_after_von_kries_foster
load mutuals_brightness_foster
load mutuals_contrast_foster

load mutuals_x4wc_foster 
Ix4wc_a=Ix4wc_a2; 
Ix4wc_b=Ix4wc_b2; 
Ix4wc_c=Ix4wc_c2;
load mutuals_x4dn_foster 
Ix4dn_a=Ix4dn_a2; 
Ix4dn_b=Ix4dn_b2; 
Ix4dn_c=Ix4dn_c2;
load mutuals_y4_foster 
Iy4_a=Iy4_a2; 
Iy4_b=Iy4_b2; 
Iy4_c=Iy4_c2;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%
%%%%  TOTAL CORRELATION
%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%
%%%
%%%    SAMPLES FOR COMPARISON IN SPACE
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Comparison in space y1o y1 x1 x3 x4dn x4wc
tamanyos=[32    32    32    32    32    16    16    16    16     8     8     8     8     3]

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

%%%%%% Seleccion optimista (18 dimensiones en low y med)
%%%%%% 

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

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

%%%
%%%  SELECCION ESPACIAL
%%%

%%%%%% Seleccion optimista (9 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (5 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

samples_space_y1o = [];
for i=1:length(y1_original_ok)
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1o = [samples_space_y1o s(seleccion_space,:)];
end

samples_space_y1 = [];
for i=1:length(y1_ok)
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1 = [samples_space_y1 s(seleccion_space,:)];
end

samples_space_x1 = [];
for i=1:length(x1_ok)
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x1 = [samples_space_x1 s(seleccion_space,:)];
end

samples_space_x3 = [];
for i=1:length(x3_ok)
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x3 = [samples_space_x3 s(seleccion_space,:)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 0;
samples_y4_focus = [];
for i = 1:length(y4_focus(1,:))
    [1 i]
    samples = scan_wavelet_for_spatial_samples(y4_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_y4_focus = [samples_y4_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

samples_x4wc_focus = [];
for i = 1:length(x4wc_focus(1,:))
    [2 i]
    samples = scan_wavelet_for_spatial_samples(x4wc_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4wc_focus = [samples_x4wc_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

x4dn_focus = x4_ok(indices_w_focus,:);
samples_x4dn_focus = [];
for i = 1:length(x4dn_focus(1,:))
    [3 i]
    samples = scan_wavelet_for_spatial_samples(x4dn_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4dn_focus = [samples_x4dn_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_12_40_foster_even_higher_dim samples_space_y1o samples_space_y1 samples_space_x1 samples_space_x3 samples_y4_focus samples_x4wc_focus samples_x4dn_focus -v7.3


%%%
%%%  SELECCION 2/3: TAMANYOS WAVELET y ESPACIAL
%%%
clear all;load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster

tam_low_select = 5;     % 5    mas lejanos  7
tam_med_select = 10;     % 10                14 
tam_hig_select = 20;    % 20                28
tam_space_select = 20;  % 20                28 

%%%
%%%  SELECCION WAVELET
%%%

%%%%%% Seleccion optimista (18 dimensiones en low y med)
%%%%%% 

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

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

%%%
%%%  SELECCION ESPACIAL
%%%

%%%%%% Seleccion optimista (9 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (5 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

samples_space_y1o = [];
for i=1:length(y1_original_ok)
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1o = [samples_space_y1o s(seleccion_space,:)];
end

samples_space_y1 = [];
for i=1:length(y1_ok)
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1 = [samples_space_y1 s(seleccion_space,:)];
end

samples_space_x1 = [];
for i=1:length(x1_ok)
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x1 = [samples_space_x1 s(seleccion_space,:)];
end

samples_space_x3 = [];
for i=1:length(x3_ok)
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x3 = [samples_space_x3 s(seleccion_space,:)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 0;
samples_y4_focus = [];
for i = 1:length(y4_focus(1,:))
    [1 i]
    samples = scan_wavelet_for_spatial_samples(y4_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_y4_focus = [samples_y4_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

samples_x4wc_focus = [];
for i = 1:length(x4wc_focus(1,:))
    [2 i]
    samples = scan_wavelet_for_spatial_samples(x4wc_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4wc_focus = [samples_x4wc_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

x4dn_focus = x4_ok(indices_w_focus,:);
samples_x4dn_focus = [];
for i = 1:length(x4dn_focus(1,:))
    [3 i]
    samples = scan_wavelet_for_spatial_samples(x4dn_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4dn_focus = [samples_x4dn_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_20_40_foster_even_higher_dim samples_space_y1o samples_space_y1 samples_space_x1 samples_space_x3 samples_y4_focus samples_x4wc_focus samples_x4dn_focus -v7.3


%%%
%%%  SELECCION 3/3: TAMANYOS WAVELET y ESPACIAL
%%%
clear all;load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster

tam_low_select = 7;     % 5    mas lejanos  7
tam_med_select = 14;     % 10                14 
tam_hig_select = 28;    % 20                28
tam_space_select = 28;  % 20                28 

%%%
%%%  SELECCION WAVELET
%%%

%%%%%% Seleccion optimista (18 dimensiones en low y med)
%%%%%% 

% seleccion low
tam_low = tam_low_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_low = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d 

% Seleccion med  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_med = tam_med_select;
tam_low = tam_med;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_med = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

% Seleccion high  (ojo!: usa el mismo nombre de arriba para variables intermedias)
tam_high = tam_hig_select;
tam_low = tam_high;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_high= indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

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

%%%
%%%  SELECCION ESPACIAL
%%%

%%%%%% Seleccion optimista (9 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low 1 ceil(tam_low/2) tam_low;
                   1 1 1 ceil(tam_low/2) ceil(tam_low/2) ceil(tam_low/2) tam_low tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%% Seleccion conservadora (5 dimensiones)
%%%%%% 
 
tam_low = tam_space_select;
indices_low_1d = reshape(1:tam_low^2,[tam_low tam_low]);
cuales_low_list = [1  tam_low ceil(tam_low/2) 1       tam_low;
                   1  1       ceil(tam_low/2) tam_low tam_low];
cuales_low_2d = zeros(tam_low,tam_low);
for i = 1:length(cuales_low_list)             
    cuales_low_2d(cuales_low_list(1,i),cuales_low_list(2,i)) = 1;    
end
seleccion_space_cons = indices_low_1d(find(cuales_low_2d==1))
% coincide con: 
cuales_low_2d.*indices_low_1d

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

samples_space_y1o = [];
for i=1:length(y1_original_ok)
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1o = [samples_space_y1o s(seleccion_space,:)];
end

samples_space_y1 = [];
for i=1:length(y1_ok)
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_y1 = [samples_space_y1 s(seleccion_space,:)];
end

samples_space_x1 = [];
for i=1:length(x1_ok)
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x1 = [samples_space_x1 s(seleccion_space,:)];
end

samples_space_x3 = [];
for i=1:length(x3_ok)
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space_select tam_space_select]);
    samples_space_x3 = [samples_space_x3 s(seleccion_space,:)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% scan wavelet for close spatial samples
conservative = 0;
samples_y4_focus = [];
for i = 1:length(y4_focus(1,:))
    [1 i]
    samples = scan_wavelet_for_spatial_samples(y4_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_y4_focus = [samples_y4_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

samples_x4wc_focus = [];
for i = 1:length(x4wc_focus(1,:))
    [2 i]
    samples = scan_wavelet_for_spatial_samples(x4wc_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4wc_focus = [samples_x4wc_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

x4dn_focus = x4_ok(indices_w_focus,:);
samples_x4dn_focus = [];
for i = 1:length(x4dn_focus(1,:))
    [3 i]
    samples = scan_wavelet_for_spatial_samples(x4dn_focus(:,i),ind_focus,tam_low_select,conservative);
    samples_x4dn_focus = [samples_x4dn_focus [samples(1,1).samples;samples(1,2).samples;samples(1,3).samples;samples(1,4).samples;samples(2,1).samples;samples(2,2).samples;samples(2,3).samples;samples(2,4).samples;samples(3,1).samples;samples(3,2).samples;samples(3,3).samples;samples(3,4).samples]];
end

save samples_spatial_tam_28_40_foster_even_higher_dim samples_space_y1o samples_space_y1 samples_space_x1 samples_space_x3 samples_y4_focus samples_x4wc_focus samples_x4dn_focus -v7.3

%%%
%%%
%%%    AND NOW, EMPIRICAL Total Correlation FROM RBIG
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 

clear all
load samples_spatial_tam_12_40_foster
%load samples_spatial_tam_12_40_foster_high_dim
%load samples_spatial_tam_12_40_foster_even_higher_dim

%  samples_space_x1               5x4352175            174087000  double              
%  samples_space_x3               5x4352175            174087000  double              
%  samples_space_y1               5x4352175            174087000  double              
%  samples_space_y1o              5x4352175            174087000  double              
%  samples_x4dn_focus            40x186300              59616000  double              
%  samples_x4wc_focus            40x186300              59616000  double              
%  samples_y4_focus  
  
PARAMS.N_lay = 600
num = round(0.85*186300);

for realiz=1:10

    ind_space = randperm(4352175);
    ind_wavel = randperm(186300);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);
 
    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_focus(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_focus(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_focus(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);

    Total_correl = [mean(Ty1o')   std(Ty1o');
    mean(Ty1')    std(Ty1');
    mean(Tx1')    std(Tx1');
    mean(Tx3')    std(Tx3');
    mean(Ty4')    std(Ty4');
    mean(Tx4wc')  std(Tx4wc');
    mean(Tx4dn')  std(Tx4dn')]

    Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
    Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
    Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
    Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
    Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
    Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

    save total_correlation_spatial_12_40_low_dimens2b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%    save total_correlation_spatial_12_40_high_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%    save total_correlation_spatial_12_40_even_higher_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load samples_spatial_tam_20_40_foster
%load samples_spatial_tam_20_40_foster_high_dim
%load samples_spatial_tam_20_40_foster_even_higher_dim

PARAMS.N_lay = 600;
num = round(0.85*82800);

for realiz=1:10

    ind_space = randperm(2282175);
    ind_wavel = randperm(82800);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_focus(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_focus(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_focus(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);

Total_correl = [mean(Ty1o')   std(Ty1o');
mean(Ty1')    std(Ty1');
mean(Tx1')    std(Tx1');
mean(Tx3')    std(Tx3');
mean(Ty4')    std(Ty4');
mean(Tx4wc')  std(Tx4wc');
mean(Tx4dn')  std(Tx4dn')]

    Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
    Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
    Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
    Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
    Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
    Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

    save total_correlation_spatial_20_40_low_dim2b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%    save total_correlation_spatial_20_40_high_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%    save total_correlation_spatial_20_40_even_higher_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load samples_spatial_tam_28_40_foster
%load samples_spatial_tam_28_40_foster_high_dim
%load samples_spatial_tam_28_40_foster_even_higher_dim
PARAMS.N_lay = 600;
num = round(0.85*20700);

for realiz=1:10

    ind_space = randperm(874575);
    ind_wavel = randperm(20700);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);
    

    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_focus(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_focus(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_focus(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
Total_correl = [mean(Ty1o')   std(Ty1o');
mean(Ty1')    std(Ty1');
mean(Tx1')    std(Tx1');
mean(Tx3')    std(Tx3');
mean(Ty4')    std(Ty4');
mean(Tx4wc')  std(Tx4wc');
mean(Tx4dn')  std(Tx4dn')]

    Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
    Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
    Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
    Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
    Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
    Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

save total_correlation_spatial_28_40_low_dim2b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%save total_correlation_spatial_28_40_high_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
%save total_correlation_spatial_28_40_even_higher_dim Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
        
end

%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (2) ..... 961 por imagen
%%%   -> 4973175 % Review here

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster

tam_low = 2;     % 3, 4
tam_space = 10;  % 13, 20
tam_por_imagen = 961;
tam_total = tam_por_imagen*5175;

samples_space_y1o_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1o_full = [samples_space_y1o_full s];
    samples_space_y1o_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [1 i] 
    end
end

samples_space_y1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1_full = [samples_space_y1_full s];
    samples_space_y1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [2 i] 
    end
end

samples_space_x1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(x1_ok(1,:))
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_x1_full = [samples_space_x1_full s];
    samples_space_x1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [3 i] 
    end
end

samples_space_x3_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    % samples_space_x3_full = [samples_space_x3_full s];
    samples_space_x3_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [4 i] 
    end
end

w = y4_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [5 i] 
    end
end
    s_H1_y4=s_H1s;    s_H2_y4=s_H2s;    s_H3_y4=s_H3s;    s_H4_y4=s_H4s;
    s_M1_y4=s_M1s;    s_M2_y4=s_M2s;    s_M3_y4=s_M3s;    s_M4_y4=s_M4s;
    s_L1_y4=s_L1s;    s_L2_y4=s_L2s;    s_L3_y4=s_L3s;    s_L4_y4=s_L4s;
    s_lows_y4=s_lows;

w = x4wc_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [6 i] 
    end
end
    s_H1_x4wc=s_H1s;    s_H2_x4wc=s_H2s;    s_H3_x4wc=s_H3s;    s_H4_x4wc=s_H4s;
    s_M1_x4wc=s_M1s;    s_M2_x4wc=s_M2s;    s_M3_x4wc=s_M3s;    s_M4_x4wc=s_M4s;
    s_L1_x4wc=s_L1s;    s_L2_x4wc=s_L2s;    s_L3_x4wc=s_L3s;    s_L4_x4wc=s_L4s;
    s_lows_x4wc=s_lows;
    
x4dn_focus = x4_ok(indices_w_focus,:);    
w = x4dn_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [7 i] 
    end
end
    s_H1_x4dn=s_H1s;    s_H2_x4dn=s_H2s;    s_H3_x4dn=s_H3s;    s_H4_x4dn=s_H4s;
    s_M1_x4dn=s_M1s;    s_M2_x4dn=s_M2s;    s_M3_x4dn=s_M3s;    s_M4_x4dn=s_M4s;
    s_L1_x4dn=s_L1s;    s_L2_x4dn=s_L2s;    s_L3_x4dn=s_L3s;    s_L4_x4dn=s_L4s;
    s_lows_x4dn=s_lows;
    
save samples_spatial_wav_tam_2_foster_full samples_space_y1o_full samples_space_y1_full samples_space_x1_full samples_space_x3_full ...
s_H1_y4 s_H2_y4 s_H3_y4 s_H4_y4 s_M1_y4 s_M2_y4 s_M3_y4 s_M4_y4 s_L1_y4 s_L2_y4 s_L3_y4 s_L4_y4 s_lows_y4 ...
s_H1_x4wc s_H2_x4wc s_H3_x4wc s_H4_x4wc s_M1_x4wc s_M2_x4wc s_M3_x4wc s_M4_x4wc s_L1_x4wc s_L2_x4wc s_L3_x4wc s_L4_x4wc s_lows_x4wc ...
s_H1_x4dn s_H2_x4dn s_H3_x4dn s_H4_x4dn s_M1_x4dn s_M2_x4dn s_M3_x4dn s_M4_x4dn s_L1_x4dn s_L2_x4dn s_L3_x4dn s_L4_x4dn s_lows_x4dn -v7.3

%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (3)  .... 784 -> 4057200
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster

tam_low = 3;     % 3, 4
tam_space = 13;  % 13, 20
tam_por_imagen = 784;
tam_total = tam_por_imagen*5175;

samples_space_y1o_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1o_full = [samples_space_y1o_full s];
    samples_space_y1o_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [1 i] 
    end
end

samples_space_y1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1_full = [samples_space_y1_full s];
    samples_space_y1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [2 i] 
    end
end

samples_space_x1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(x1_ok(1,:))
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_x1_full = [samples_space_x1_full s];
    samples_space_x1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [3 i] 
    end
end

samples_space_x3_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    % samples_space_x3_full = [samples_space_x3_full s];
    samples_space_x3_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [4 i] 
    end
end

w = y4_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [5 i] 
    end
end
    s_H1_y4=s_H1s;    s_H2_y4=s_H2s;    s_H3_y4=s_H3s;    s_H4_y4=s_H4s;
    s_M1_y4=s_M1s;    s_M2_y4=s_M2s;    s_M3_y4=s_M3s;    s_M4_y4=s_M4s;
    s_L1_y4=s_L1s;    s_L2_y4=s_L2s;    s_L3_y4=s_L3s;    s_L4_y4=s_L4s;
    s_lows_y4=s_lows;

w = x4wc_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [6 i] 
    end
end
    s_H1_x4wc=s_H1s;    s_H2_x4wc=s_H2s;    s_H3_x4wc=s_H3s;    s_H4_x4wc=s_H4s;
    s_M1_x4wc=s_M1s;    s_M2_x4wc=s_M2s;    s_M3_x4wc=s_M3s;    s_M4_x4wc=s_M4s;
    s_L1_x4wc=s_L1s;    s_L2_x4wc=s_L2s;    s_L3_x4wc=s_L3s;    s_L4_x4wc=s_L4s;
    s_lows_x4wc=s_lows;
    
x4dn_focus = x4_ok(indices_w_focus,:);
w = x4dn_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [7 i] 
    end
end
    s_H1_x4dn=s_H1s;    s_H2_x4dn=s_H2s;    s_H3_x4dn=s_H3s;    s_H4_x4dn=s_H4s;
    s_M1_x4dn=s_M1s;    s_M2_x4dn=s_M2s;    s_M3_x4dn=s_M3s;    s_M4_x4dn=s_M4s;
    s_L1_x4dn=s_L1s;    s_L2_x4dn=s_L2s;    s_L3_x4dn=s_L3s;    s_L4_x4dn=s_L4s;
    s_lows_x4dn=s_lows;
    
save samples_spatial_wav_tam_3_foster_full samples_space_y1o_full samples_space_y1_full samples_space_x1_full samples_space_x3_full ...
s_H1_y4 s_H2_y4 s_H3_y4 s_H4_y4 s_M1_y4 s_M2_y4 s_M3_y4 s_M4_y4 s_L1_y4 s_L2_y4 s_L3_y4 s_L4_y4 s_lows_y4 ...
s_H1_x4wc s_H2_x4wc s_H3_x4wc s_H4_x4wc s_M1_x4wc s_M2_x4wc s_M3_x4wc s_M4_x4wc s_L1_x4wc s_L2_x4wc s_L3_x4wc s_L4_x4wc s_lows_x4wc ...
s_H1_x4dn s_H2_x4dn s_H3_x4dn s_H4_x4dn s_M1_x4dn s_M2_x4dn s_M3_x4dn s_M4_x4dn s_L1_x4dn s_L2_x4dn s_L3_x4dn s_L4_x4dn s_lows_x4dn -v7.3

%% 
%%%
%%%   GATHERING DATA WITH NO SPATIAL SUBSAMPLING  (4)  .... 441 -> 2282175
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
load all_data_OK_foster_y1o_y1_x1_x3_y4_x4dn_x4wc
load marginal_statistics_foster

tam_low = 4;     % 3, 4
tam_space = 20;  % 13, 20
tam_por_imagen = 441;
tam_total = tam_por_imagen*5175;

samples_space_y1o_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_original_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1o_full = [samples_space_y1o_full s];
    samples_space_y1o_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [1 i] 
    end
end

samples_space_y1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(y1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_y1_full = [samples_space_y1_full s];
    samples_space_y1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [2 i] 
    end
end

samples_space_x1_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(x1_ok(1,:))
    im = reshape(x1_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    %samples_space_x1_full = [samples_space_x1_full s];
    samples_space_x1_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [3 i] 
    end
end

samples_space_x3_full = zeros(tam_space*tam_space,tam_total);
for i=1:length(y1_original_ok(1,:))
    im = reshape(x3_ok(:,i),[40 40]);
    s = im2col(im,[tam_space tam_space]);
    % samples_space_x3_full = [samples_space_x3_full s];
    samples_space_x3_full(:,(i-1)*tam_por_imagen+1:i*tam_por_imagen) = s;
    if mod(i,50)==0
       [4 i] 
    end
end

w = y4_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [5 i] 
    end
end
    s_H1_y4=s_H1s;    s_H2_y4=s_H2s;    s_H3_y4=s_H3s;    s_H4_y4=s_H4s;
    s_M1_y4=s_M1s;    s_M2_y4=s_M2s;    s_M3_y4=s_M3s;    s_M4_y4=s_M4s;
    s_L1_y4=s_L1s;    s_L2_y4=s_L2s;    s_L3_y4=s_L3s;    s_L4_y4=s_L4s;
    s_lows_y4=s_lows;

w = x4wc_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [6 i] 
    end
end
    s_H1_x4wc=s_H1s;    s_H2_x4wc=s_H2s;    s_H3_x4wc=s_H3s;    s_H4_x4wc=s_H4s;
    s_M1_x4wc=s_M1s;    s_M2_x4wc=s_M2s;    s_M3_x4wc=s_M3s;    s_M4_x4wc=s_M4s;
    s_L1_x4wc=s_L1s;    s_L2_x4wc=s_L2s;    s_L3_x4wc=s_L3s;    s_L4_x4wc=s_L4s;
    s_lows_x4wc=s_lows;
    
x4dn_focus = x4_ok(indices_w_focus,:);
w = x4dn_focus;
s_H1s=[];s_H2s=[];s_H3s=[];s_H4s=[];s_M1s=[];s_M2s=[];s_M3s=[];s_M4s=[];s_L1s=[];s_L2s=[];s_L3s=[];s_L4s=[];s_lows=[];
for i=1:length(w(1,:))
    [s_H1,s_H2,s_H3,s_H4,s_M1,s_M2,s_M3,s_M4,s_L1,s_L2,s_L3,s_L4,s_low] = scan_wavelet_for_spatial_samples_full(w(:,i),ind_focus,tam_low);
    s_H1s=[s_H1s s_H1];    s_H2s=[s_H2s s_H2];    s_H3s=[s_H3s s_H3];    s_H4s=[s_H4s s_H4];
    s_M1s=[s_M1s s_M1];    s_M2s=[s_M2s s_M2];    s_M3s=[s_M3s s_M3];    s_M4s=[s_M4s s_M4];
    s_L1s=[s_L1s s_L1];    s_L2s=[s_L2s s_L2];    s_L3s=[s_L3s s_L3];    s_L4s=[s_L4s s_L4];
    s_lows=[s_lows s_low];
    if mod(i,50)==0
       [7 i] 
    end
end
    s_H1_x4dn=s_H1s;    s_H2_x4dn=s_H2s;    s_H3_x4dn=s_H3s;    s_H4_x4dn=s_H4s;
    s_M1_x4dn=s_M1s;    s_M2_x4dn=s_M2s;    s_M3_x4dn=s_M3s;    s_M4_x4dn=s_M4s;
    s_L1_x4dn=s_L1s;    s_L2_x4dn=s_L2s;    s_L3_x4dn=s_L3s;    s_L4_x4dn=s_L4s;
    s_lows_x4dn=s_lows;
    
save samples_spatial_wav_tam_4_foster_full samples_space_y1o_full samples_space_y1_full samples_space_x1_full samples_space_x3_full ...
s_H1_y4 s_H2_y4 s_H3_y4 s_H4_y4 s_M1_y4 s_M2_y4 s_M3_y4 s_M4_y4 s_L1_y4 s_L2_y4 s_L3_y4 s_L4_y4 s_lows_y4 ...
s_H1_x4wc s_H2_x4wc s_H3_x4wc s_H4_x4wc s_M1_x4wc s_M2_x4wc s_M3_x4wc s_M4_x4wc s_L1_x4wc s_L2_x4wc s_L3_x4wc s_L4_x4wc s_lows_x4wc ...
s_H1_x4dn s_H2_x4dn s_H3_x4dn s_H4_x4dn s_M1_x4dn s_M2_x4dn s_M3_x4dn s_M4_x4dn s_L1_x4dn s_L2_x4dn s_L3_x4dn s_L4_x4dn s_lows_x4dn -v7.3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%

clear all
load samples_spatial_wav_tam_2_foster_full
h = genpath('/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/2017_RBIG');
addpath(h)


PARAMS.N_lay = 600;
num = round(0.85*82800);
samples_y4 = [s_H1_y4;s_H2_y4;s_H3_y4;s_H4_y4;s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn = [s_H1_x4dn;s_H2_x4dn;s_H3_x4dn;s_H4_x4dn;s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc = [s_H1_x4wc;s_H2_x4wc;s_H3_x4wc;s_H4_x4wc;s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

samples_y4_ml = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn_ml = [s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc_ml = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

for realiz=1:10

    ind_space = randperm(4973175);
    ind_wavel = randperm(82800);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o_full(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1_full(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1_full(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz];
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3_full(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_ml(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_ml(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_ml(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);

Total_correl = [mean(Ty1o')   std(Ty1o');
mean(Ty1')    std(Ty1');
mean(Tx1')    std(Tx1');
mean(Tx3')    std(Tx3');
mean(Ty4')    std(Ty4');
mean(Tx4wc')  std(Tx4wc');
mean(Tx4dn')  std(Tx4dn')]

Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

save total_correlation_foster_full_size_2b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  4057200
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  46575

clear all
load samples_spatial_wav_tam_3_foster_full

PARAMS.N_lay = 600;
num = round(0.85*46575);
samples_y4 = [s_H1_y4;s_H2_y4;s_H3_y4;s_H4_y4;s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn = [s_H1_x4dn;s_H2_x4dn;s_H3_x4dn;s_H4_x4dn;s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc = [s_H1_x4wc;s_H2_x4wc;s_H3_x4wc;s_H4_x4wc;s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

samples_y4_ml = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn_ml = [s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc_ml = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

for realiz=1:10

    ind_space = randperm(4057200);
    ind_wavel = randperm(46575);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o_full(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1_full(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1_full(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3_full(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_ml(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_ml(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_ml(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);

Total_correl = [mean(Ty1o')   std(Ty1o');
mean(Ty1')    std(Ty1');
mean(Tx1')    std(Tx1');
mean(Tx3')    std(Tx3');
mean(Ty4')    std(Ty4');
mean(Tx4wc')  std(Tx4wc');
mean(Tx4dn')  std(Tx4dn')]

Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

save total_correlation_foster_full_size_3b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2282175
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 20700

clear all
load samples_spatial_wav_tam_4_foster_full

PARAMS.N_lay = 600;
num = round(0.85*20700);
samples_y4 = [s_H1_y4;s_H2_y4;s_H3_y4;s_H4_y4;s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn = [s_H1_x4dn;s_H2_x4dn;s_H3_x4dn;s_H4_x4dn;s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc = [s_H1_x4wc;s_H2_x4wc;s_H3_x4wc;s_H4_x4wc;s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

samples_y4_ml = [s_M1_y4;s_M2_y4;s_M3_y4;s_M4_y4;s_L1_y4;s_L2_y4;s_L3_y4;s_L4_y4;s_lows_y4];
samples_x4dn_ml = [s_M1_x4dn;s_M2_x4dn;s_M3_x4dn;s_M4_x4dn;s_L1_x4dn;s_L2_x4dn;s_L3_x4dn;s_L4_x4dn;s_lows_x4dn];
samples_x4wc_ml = [s_M1_x4wc;s_M2_x4wc;s_M3_x4wc;s_M4_x4wc;s_L1_x4wc;s_L2_x4wc;s_L3_x4wc;s_L4_x4wc;s_lows_x4wc];

for realiz=1:10

    ind_space = randperm(2282175);
    ind_wavel = randperm(20700);

    indices_space = ind_space(1:num);
    indices_wavel = ind_wavel(1:num);

    [1 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1o_full(:,indices_space),PARAMS);
    Ty1o(realiz) = PARAMSo.MI
    Ty1o_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [2 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_y1_full(:,indices_space),PARAMS);
    Ty1(realiz) = PARAMSo.MI
    Ty1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [3 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x1_full(:,indices_space),PARAMS);
    Tx1(realiz) = PARAMSo.MI
    Tx1_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [4 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_space_x3_full(:,indices_space),PARAMS);
    Tx3(realiz) = PARAMSo.MI
    Tx3_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
    [5 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_y4_ml(:,indices_wavel),PARAMS);
    Ty4(realiz) = PARAMSo.MI
    Ty4_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [6 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4wc_ml(:,indices_wavel),PARAMS);
    Tx4wc(realiz) = PARAMSo.MI
    Tx4wc_conv(realiz,:) = integra_convergencia600(PARAMSo);

    [7 realiz]
    [datT Trans PARAMSo] = RBIG_2017(samples_x4dn_ml(:,indices_wavel),PARAMS);
    Tx4dn(realiz) = PARAMSo.MI
    Tx4dn_conv(realiz,:) = integra_convergencia600(PARAMSo);
    
Total_correl = [mean(Ty1o')   std(Ty1o');
mean(Ty1')    std(Ty1');
mean(Tx1')    std(Tx1');
mean(Tx3')    std(Tx3');
mean(Ty4')    std(Ty4');
mean(Tx4wc')  std(Tx4wc');
mean(Tx4dn')  std(Tx4dn')]

Ty1o_converg = [mean(Ty1o_conv);std(Ty1o_conv)]; 
Ty1_converg = [mean(Ty1_conv); std(Ty1_conv)];
Tx1_converg = [mean(Tx1_conv); std(Tx1_conv)];
Tx3_converg = [mean(Tx3_conv); std(Tx3_conv)];Ty4_converg = [mean(Ty4_conv); std(Ty4_conv)];
Tx4wc_converg = [mean(Tx4wc_conv); std(Tx4wc_conv)];
Tx4dn_converg = [mean(Tx4dn_conv); std(Tx4dn_conv)];

save total_correlation_foster_full_size_4b Total_correl Ty1o Ty1 Tx1 Tx3 Ty4 Tx4wc Tx4dn Ty1o_converg Ty1_converg Tx1_converg Tx3_converg Ty4_converg Tx4wc_converg Tx4dn_converg
    
end

%%%
%%% ANALYSIS OF TOTAL CORRELATION RESULTS (computed above) IS DONE IN TWO STEPS
%%%
%%%      * Given the fact that the redundant wavelet transform introducs certain redundancy ;-)
%%%        I estimate the resundancy introduced by looking at the T of Gaussian white noise
%%%        in the wavelet domain. This T should be the reference to compare
%%%        with in wavelet-like domains (i.e. we should subtract this)
%%%
%%%        I estimate that in ..... T_in_redudndant_transform.m 
%%%                                        total_correlation_spatial_noise_12_40_low_dim;
%%%                                        total_correlation_foster_full_size_2b
%%%
%%%      * Then the results are analyzed in .... summary_of_T_results.m
%%%
%%%              - This loads and represents the empirical results computed with RBIG
%%%                and computes the theoretical results.
%%%