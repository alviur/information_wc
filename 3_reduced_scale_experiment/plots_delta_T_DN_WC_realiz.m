x0 = [];
y2 = [];
x2 = [];

x0wc = [];
y2wc = [];
x2wc = [];

%%
%%  GATHER SAMPLES AND VARIATIONS OF TOTAL CORRELATION
%%

for ii=1:10
    ii
    % name = ['_N5e6_realiz_',num2str(ii)]          % 1:16
    % name = ['_N5e6_01_realiz_',num2str(ii)]      % 1:10
    %name = ['_N5e6_25_01_realiz_',num2str(ii)]      % 1:10
    name = ['_N5e6_25_wide_realiz_',num2str(ii)]      % 1:10    
    % name = ['_foster_N7e6_01_realiz_',num2str(ii)]  % 1:6
    load(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/3D_example_jesus/redundancy_reduction_small',name])
    delta_I_rbig_10000(:,:,ii) = DELTA_I_rep;
    delta_I_stud_10000(:,:,ii) = Delta_I_local;
    delta_H_margi_10000(:,:,ii)= Delta_H_local;
    delta_I_stud_g_10000(:,:,ii) = Delta_I_global;
    delta_H_margi_g_10000(:,:,ii)= Delta_H_global;
    
    delta_I_rbigwc_10000(:,:,ii) = DELTA_I_repwc;
    delta_I_studwc_10000(:,:,ii) = Delta_I_localwc;
    delta_H_margiwc_10000(:,:,ii)= Delta_H_localwc;
    delta_I_stud_gwc_10000(:,:,ii) = Delta_I_globalwc;
    delta_H_margi_gwc_10000(:,:,ii)= Delta_H_globalwc;
    ElogdetJ_10000(:,:,ii) = Delta_I_local - Delta_H_local;
    ElogdetJwc_10000(:,:,ii) = Delta_I_localwc - Delta_H_localwc;
    histo_10000(:,:,ii) = H;

    tam0 = size(samples);
    tam1 = size(samples_transf_1);
    tam2 = size(samples_transf_2);
    for i=1:tam1(1)
        for j=1:tam1(2)
            x0 = [x0 samples_transf_0(i,j).y1];
            y2 = [y2 samples_transf_1(i,j).y2];            
            x2 = [x2 samples_transf_2(i,j).x2]; 
            
            x0wc = [x0wc real(samples_transf_0wc(i,j).y1)];
            y2wc = [y2wc samples_transf_1wc(i,j).y2];            
            x2wc = [x2wc samples_transf_2wc(i,j).x2];                        
        end
    end
end

%% SAMPLES TO REPRESENT

indi = randperm(length(x2(1,:)));
indi = indi(1:5000);

y11samples = sort([min(x0(1,:)) x0(1,indi) max(x0(1,:))]);
y12samples = sort([min(x0(2,:)) x0(2,indi) max(x0(2,:))]);
y13samples = sort([min(x0(3,:)) x0(3,indi) max(x0(3,:))]);

y21samples = sort([min(y2(1,:)) y2(1,indi) max(y2(1,:))]);
y22samples = sort([min(y2(2,:)) y2(2,indi) max(y2(2,:))]);
y23samples = sort([min(y2(3,:)) y2(3,indi) max(y2(3,:))]);

x21samples = sort([min(x2(1,:)) x2(1,indi) max(x2(1,:))]);
x22samples = sort([min(x2(2,:)) x2(2,indi) max(x2(2,:))]);
x23samples = sort([min(x2(3,:)) x2(3,indi) max(x2(3,:))]);

%%

y11samples_wc = sort([min(x0wc(1,:)) x0wc(1,indi) max(x0wc(1,:))]);
y12samples_wc = sort([min(x0wc(2,:)) x0wc(2,indi) max(x0wc(2,:))]);
y13samples_wc = sort([min(x0wc(3,:)) x0wc(3,indi) max(x0wc(3,:))]);

y21samples_wc = sort([min(y2wc(1,:)) y2wc(1,indi) max(y2wc(1,:))]);
y22samples_wc = sort([min(y2wc(2,:)) y2wc(2,indi) max(y2wc(2,:))]);
y23samples_wc = sort([min(y2wc(3,:)) y2wc(3,indi) max(y2wc(3,:))]);

x21samples_wc = sort([min(x2wc(1,:)) x2wc(1,indi) max(x2wc(1,:))]);
x22samples_wc = sort([min(x2wc(2,:)) x2wc(2,indi) max(x2wc(2,:))]);
x23samples_wc = sort([min(x2wc(3,:)) x2wc(3,indi) max(x2wc(3,:))]);

%% MEAN VARIATIONS OF TOTAL CORRELATION
   
     delta_I_RBIG_10000 = mean(delta_I_rbig_10000,3);
std_delta_I_RBIG_10000 =  std(delta_I_rbig_10000,0,3);

    delta_I_STUD_10000 = mean(delta_I_stud_10000,3);
std_delta_I_STUD_10000 =  std(delta_I_stud_10000,0,3);
    delta_H_Margi_10000 = mean(delta_H_margi_10000,3);
std_delta_H_Margi_10000 =  std(delta_H_margi_10000,0,3);
    delta_I_STUD_g_10000 = mean(delta_I_stud_g_10000,3);
std_delta_I_STUD_g_10000 =  std(delta_I_stud_g_10000,0,3);
       E_log_detJ_10000 = mean(ElogdetJ_10000,3);
   std_E_log_detJ_10000 =  std(ElogdetJ_10000,0,3);
   
    delta_I_RBIGwc_10000 = mean(delta_I_rbigwc_10000,3);
std_delta_I_RBIGwc_10000 =  std(delta_I_rbigwc_10000,0,3);
   
    delta_I_STUDwc_10000 = mean(delta_I_studwc_10000,3);
std_delta_I_STUDwc_10000 =  std(delta_I_studwc_10000,0,3);
    delta_H_Margiwc_10000 = mean(delta_H_margiwc_10000,3);
std_delta_H_Margiwc_10000 =  std(delta_H_margiwc_10000,0,3);
    delta_I_STUD_gwc_10000 = mean(delta_I_stud_gwc_10000,3);
std_delta_I_STUD_gwc_10000 =  std(delta_I_stud_gwc_10000,0,3);
       E_log_detJwc_10000 = mean(ElogdetJwc_10000,3);
   std_E_log_detJcw_10000 =  std(ElogdetJwc_10000,0,3);

       HISTO_10000 = mean(histo_10000,3);
   std_HISTO_10000 =  std(histo_10000,0,3);  
   
%% ----------------------------------------------------------
   
nbins = round(sqrt(length(x0(1,:)))/100);
[px01bien,lum1] = hist(x0(1,:),nbins);
[px02bien,lum2] = hist(x0(2,:),nbins);
[px03bien,lum3] = hist(x0(3,:),nbins);

[py21bien,brightbien] = hist(y2(1,:),nbins);
[py22bien,c_lowbien] = hist(y2(2,:),nbins);
[py23bien,c_highbien] = hist(y2(3,:),nbins);

[px21bien,brightbienNL] = hist(x2(1,:),nbins);
[px22bien,c_lowbienNL] = hist(x2(2,:),nbins);
[px23bien,c_highbienNL] = hist(x2(3,:),nbins);

[px01bienwc,lum1wc] = hist(x0wc(1,:),nbins);
[px02bienwc,lum2wc] = hist(x0wc(2,:),nbins);
[px03bienwc,lum3wc] = hist(x0wc(3,:),nbins);

[py21bienwc,brightbienwc] = hist(y2wc(1,:),nbins);
[py22bienwc,c_lowbienwc] = hist(y2wc(2,:),nbins);
[py23bienwc,c_highbienwc] = hist(y2wc(3,:),nbins);

[px21bienwc,brightbienNLwc] = hist(x2wc(1,:),nbins);
[px22bienwc,c_lowbienNLwc] = hist(x2wc(2,:),nbins);
[px23bienwc,c_highbienNLwc] = hist(x2wc(3,:),nbins);

C=contrasts;
L=luminances;

lw=2;
color = [0.6 0.8 1];

figure(1),subplot(121),t=mesh(C,L,delta_I_RBIG_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaT DN ( RBIG )'),zlabel('\DeltaT (bits)'),axis([0 0.85 0 0.85 -0.25 7])
hold on,t=mesh(C,L,std_delta_I_RBIG_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',0.5*color,'linewidth',lw)
figure(1),subplot(122),t=mesh(C,L,delta_I_STUD_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaT DN ( Theoret. )'),zlabel('\DeltaT (bits)'),axis([0 0.85 0 0.85 -0.25 7])
hold on,t=mesh(C,L,std_delta_I_STUD_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',0.5*color,'linewidth',lw)
% figure(1),subplot(133),t=mesh(C,L,delta_I_STUD_g_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
% view([122 40]),title('\DeltaT DN ( Theoret., n = 10000 )')
set(gcf,'color',[1 1 1])

figure(2),subplot(121),t=mesh(C,L,delta_I_RBIGwc_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaT WC ( RBIG )'),zlabel('\DeltaT (bits)'),axis([0 0.85 0 0.85 -0.25 4.5])
hold on,t=mesh(C,L,std_delta_I_RBIGwc_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',0.5*color,'linewidth',lw)
figure(2),subplot(122),t=mesh(C,L,delta_I_STUDwc_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaT WC ( Theoret. )'),zlabel('\DeltaT (bits)'),axis([0 0.85 0 0.85 -0.25 4.5])
hold on,t=mesh(C,L,std_delta_I_STUDwc_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',0.5*color,'linewidth',lw)
%figure(2),subplot(133),t=mesh(C,L,delta_I_STUD_gwc_10000),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
%view([122 40]),title('\DeltaT WC ( Theoret. )')
set(gcf,'color',[1 1 1])

P = HISTO_10000;
P = P/(sum(sum(P))*(C(2)-C(1))*(L(2)-L(1)));
figure(3),t=mesh(C,L,sqrt(P)),xlabel({['C_{RMSE} = 2^{1/2} \sigma / L'],['(Contrast)']}),ylabel({['L'],['(Average Norm. Luminance)']}),set(t,'facecolor',color,'linewidth',lw)
zlabel('PDF^{1/2}')
view([122 40]),title({'PDF of Natural Images','in the Luminance / Contrast plane'}),
axis([0 0.85 0 0.85 0 (0.1/((C(2)-C(1))*(L(2)-L(1)))).^0.5])
set(gcf,'color',[1 1 1])

origin = 0;
figure(111),subplot(121),t=mesh(C,L,E_log_detJ_10000+origin),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('E[log_2|\nablaS|]    DN'),axis([0 0.85 0 0.85 -12 12.3])
figure(111),subplot(122),t=mesh(C,L,delta_H_Margi_10000-origin),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaH   DN'),axis([0 0.85 0 0.85 -12 12.3])

%ElogJ2 = delta_I_STUDwc_10000 - delta_H_Margiwc_10000;
%figure(112),subplot(131),t=mesh(C,L,ElogJ2),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
%view([122 40]),title('E[log_2|\nablaS|]    WC (subt of averages)'),axis([0 0.85 0 0.85 -8 10])
origin = 0;
figure(112),subplot(121),t=mesh(C,L,E_log_detJwc_10000+origin),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('E[log_2|\nablaS|]    WC'),axis([0 0.85 0 0.85 -6.5 8])
figure(112),subplot(122),t=mesh(C,L,delta_H_Margiwc_10000-origin),xlabel('C'),ylabel('L'),set(t,'facecolor',color,'linewidth',lw)
view([122 40]),title('\DeltaH   WC'),axis([0 0.85 0 0.85 -6.5 8])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% figure(1),xlabel('z^1_1 Normalized Luminance in pixel 1')
% figure(2),xlabel('z^1_2, y^1_3 Normalized Luminance in pixels 2, 3')
% figure(3),xlabel('z^2_1 Brigthness')
% figure(4),xlabel('z^2_2, z^2_3 Low and high freq. components (contrast)')
% figure(5),xlabel('x^2_1 Response to Brigthness')
% figure(6),xlabel('x^2_2, x^2_3 Resp. to low and high freq. components')

%% LAS CURVAS DE RESPUESTAS SON ANODINAS

figure,
subplot(131),plot([0 y21samples],[0 x21samples],'linewidth',lw,'color',[1 0 0]),title('Zero-frequency'),xlabel('z^2_1 (Brightness)'),ylabel('Response x^2_1')
axis([0 2.5 0 0.8])
subplot(132),plot(y22samples,x22samples,'linewidth',lw,'color',[0 0.6 0]),title('Low-frequency'),xlabel('z^2_2 (low-freq. contrast)'),ylabel('Response x^2_2')
subplot(133),plot(y23samples,x23samples,'linewidth',lw,'color',[0 0 1]),title('High-frequency'),xlabel('z^2_3 (High-freq. contrast)'),ylabel('Response x^2_3')
set(gcf,'color',[1 1 1 ])

figure,
subplot(131),plot([0 y21samples_wc],[0 x21samples_wc],'linewidth',lw,'color',[1 0 0]),title('Zero-frequency'),xlabel('z^2_1 (Brightness)'),ylabel('Response x^2_1')
axis([0 2.5 0 0.8])
subplot(132),plot(y22samples_wc,x22samples_wc,'linewidth',lw,'color',[0 0.6 0]),title('Low-frequency'),xlabel('z^2_2 (low-freq. contrast)'),ylabel('Response x^2_2')
subplot(133),plot(y23samples_wc,x23samples_wc,'linewidth',lw,'color',[0 0 1]),title('High-frequency'),xlabel('z^2_3 (High-freq. contrast)'),ylabel('Response x^2_3')
set(gcf,'color',[1 1 1 ])

%%%% LAS MARGINALES SALEN FEAS :-(

figure,loglog(lum1,px01bien,'linewidth',lw,'color',[1 0 0]),title('PDF of luminance')
hold on,loglog(lum2,px02bien,'linewidth',lw,'color',[0 0.6 0]),
hold on,loglog(lum3,px03bien,'linewidth',lw,'color',[0 0 1]),

figure,semilogy(brightbien,py21bien,'linewidth',lw,'color',[1 0 0]),title('PDF of brightness, \theta_2')
figure,semilogy(c_lowbien,py22bien,'linewidth',lw,'color',[0 0.6 0]),title('PDF of low-freq. contrast, \theta_2')
hold on,semilogy(c_highbien,py23bien,'linewidth',lw,'color',[0 0 1]),title('PDF of high-freq. contrast, \theta_2')

figure,semilogy(brightbienNL,px21bien,'linewidth',lw,'color',[1 0 0]),title('PDF of nonlin-brightness')
figure,semilogy(c_lowbienNL,px22bien,'linewidth',lw,'color',[0 0.6 0]),title('PDF of low-freq. nonlin contrast')
hold on,semilogy(c_highbienNL,px23bien,'linewidth',lw,'color',[0 0 1]),title('PDF of high-freq. nonlin contrast')


figure,loglog(lum1wc,px01bienwc,'linewidth',lw,'color',[1 0 0]),title('PDF of luminance WC')
hold on,loglog(lum2wc,px02bienwc,'linewidth',lw,'color',[0 0.6 0]),
hold on,loglog(lum3wc,px03bienwc,'linewidth',lw,'color',[0 0 1]),

figure,semilogy(brightbienwc,py21bienwc,'linewidth',lw,'color',[1 0 0]),title('PDF of brightness, WC')
figure,semilogy(c_lowbienwc,py22bienwc,'linewidth',lw,'color',[0 0.6 0]),title('PDF of low-freq. contrast, WC')
hold on,semilogy(c_highbienwc,py23bienwc,'linewidth',lw,'color',[0 0 1]),title('PDF of high-freq. contrast, WC')

figure,semilogy(brightbienNLwc,px21bienwc,'linewidth',lw,'color',[1 0 0]),title('PDF of nonlin-brightness WC')
figure,semilogy(c_lowbienNLwc,px22bienwc,'linewidth',lw,'color',[0 0.6 0]),title('PDF of low-freq. nonlin contrast WC')
hold on,semilogy(c_highbienNLwc,px23bienwc,'linewidth',lw,'color',[0 0 1]),title('PDF of high-freq. nonlin contrast WC')


cada = 3000;
figure(1),plot3(x0(1,1:cada:end),x0(2,1:cada:end),x0(3,1:cada:end),'b.'),box on,grid on
xlabel({['        r^1_1'],['Norm. Lumin.'],['    in pixel 1']}),
ylabel({['        r^1_2'],['Norm. Lumin.'],['    in pixel 2']}),
zlabel({['        r^1_3'],['Norm. Lumin.'],['    in pixel 3']}),
title('Luminance in the Spatial Domain'),set(gcf,'color',[1 1 1])
axis([0 1.9 0 1.9 0 1.9]),view([-17 28])
ax = gca;ax.BoxStyle = 'full';
figure(2),plot3(y2(1,1:cada:end),y2(2,1:cada:end),y2(3,1:cada:end),'b.'),box on,grid on
xlabel({['       r^2_1'],['DC Brightness'],['(zero frequency)']}),
ylabel({['       r^2_2'],['   Low-freq. Contrast']}),
zlabel({['       r^2_3'],['High-freq. Contrast']}),
title('Frequency decomposition of Brightness'),set(gcf,'color',[1 1 1])
axis([0 2.5 -0.5 0.5 -0.3 0.3]),view([-47 31])
ax = gca;ax.BoxStyle = 'full';
figure(3),plot3(x2(1,1:cada:end),x2(2,1:cada:end),x2(3,1:cada:end),'b.'),box on,grid on
xlabel({['     x^2_1'],['Resp. to Bright.']}),
ylabel({['x^2_2'],['      Resp. to Low-freq.']}),
zlabel({['x^2_3'],['Resp. to High-freq.']}),
title({'Responses to Brightness/Contrast';'(Divisive Normalization)'}),set(gcf,'color',[1 1 1])
%axis([0.5 0.78 -0.1 0.1 -0.015 0.015]),
ax = gca;ax.BoxStyle = 'full';
view([104 10])

cada = 3000;
figure(4),plot3(x0wc(1,1:cada:end),x0wc(2,1:cada:end),x0wc(3,1:cada:end),'b.'),box on,grid on
xlabel({['        z^1_1'],['Norm. Lumin.'],['         in pixel 1']}),
ylabel({['        z^1_2'],['Norm. Luminance'],['    in pixel 2']}),
zlabel({['        z^1_3'],['Norm. Luminance'],['    in pixel 3']}),
title('Luminance in the Spatial Domain'),set(gcf,'color',[1 1 1])
axis([0 1.9 0 1.9 0 1.9]),view([-17 28])
ax = gca;ax.BoxStyle = 'full';
figure(5),plot3(y2wc(1,1:cada:end),y2wc(2,1:cada:end),y2wc(3,1:cada:end),'b.'),box on,grid on
xlabel({['             z^2_1'],['Average Brightness'],['(zero frequency)']}),
ylabel({['             z^2_2'],['   Low-freq. Contrast']}),
zlabel({['             z^2_3'],['High-freq. Contrast']}),
title('Frequency decomposition of Brightness'),set(gcf,'color',[1 1 1])
ax = gca;ax.BoxStyle = 'full';
%axis([0 2.5 -0.5 0.5 -0.3 0.3]),view([-47 31])
figure(6),plot3(x2wc(1,1:cada:end),x2wc(2,1:cada:end),x2wc(3,1:cada:end),'b.'),box on,grid on
xlabel({['     x^2_1'],['Resp. to Bright.']}),
ylabel({['x^2_2'],['      Resp. to Low-freq.']}),
zlabel({['x^2_3'],['Resp. to High-freq.']}),
title({'Responses to Brightness/Contrast';'(Wilson-Cowan)'}),set(gcf,'color',[1 1 1])
axis([0 2 -0.8 0.8 -0.5 0.5]),
ax = gca;ax.BoxStyle = 'full';
view([104 10])