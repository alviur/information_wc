function Redundancy_Reduction_small_DN_WC_fast(param,num,Nfast,name)

%%
%%
%%   EXAMPLES 1 and 2: ADAPTIVE SENSITIVITY AND REDUNDANCY REDUCTION
%%
%%

%%%%%% Mini model (works as expected!)

% General trends
%
%   - Sensitivity decreases with luminance and this has three contributions
%     (y11, y21 and x21)
%   - Sensitivity decreases with contrast (y2i, x2i)
%   - Sensitivity increases with CSF gain (depends on the product)
%   - Luminance dependence decreases with gamma1 = 1 (but it doesnot disappears with gamma1 = 1)
%   - Contrast dependence increases by reducing b
%   
    
addpath(genpath('/media/disk/vista/Papers/2017_Information_Flow/2017_RBIG/'))
%addpath(genpath('/media/disk/vista/Papers/2017_Information_Flow/BioMultiLayer_L_NL_color/'))
addpath(genpath('/media/disk/vista/Papers/IMAGING/fMRI_explora/Vistalab/'))
% addpath(genpath('/media/disk/vista/Papers/2018_RBIG_IT_measures/experimentos/Jesus_updated/'))
addpath(genpath('/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/3D_example_jesus/'))


load images_80  
im = [im1 im2 im3]/256;
x = im2col(im,[1 3],'sliding');
xx=x;
[y1,x1,y2,x2,J1,J2] = stabilized_DN_param(xx,param,0);
x2_a = mean(abs(x2)')';
% x2_a = [1.1210 0.0212 0.0063]'; 
g=0.4;
k=[1/8 1/15 1/60].^g;
k=0.75*k/max(k);
k = diag(k);
% k = [0.7500    0.5833    0.3350]
deltat=1e-5;
Nit=500;

clear x xx

close all
FIG = 10;

%% VAN HATEREN
%load /media/disk/vista/BBDD/Image_Statistic/Van_Hateren/VANH_subsampled_images
load /media/disk/databases/BBDD_video_image/Image_Statistic/Van_Hateren/VANH_subsampled_images

imag = randperm(4167);
xx = [];
for i=1:400
    im = images(imag(i)).vanH;
    x = im2col(im,[1 3],'sliding'); 
    rand_indices=randperm(length(x(1,:)));
    xx = [xx x(:,rand_indices(1:round(num/400)))];
end
x = xx;clear xx images
x = x + 1e-10*randn(3,length(x(1,:)));

contrasts = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6];
luminances = [0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7];

contrasts = linspace(0,1.1,11);
luminances = linspace(0,2.5,10);

for l = 1:length(luminances)
    for m = 1:length(contrasts)
        samples(l,m).x = [];     
    end
end

%for i=1:length(x(1,:))
%    xx = x(:,i);
%    Y = mean(xx);
%    C = sqrt(2)*std(xx)/Y;
%    l = find( abs(luminances-Y) == min(abs(luminances-Y)) );
%    m = find( abs(contrasts-C) == min(abs(contrasts-C)) );
%    samples(l,m).x = [samples(l,m).x xx];
%end

Y = mean(x,1);
C = sqrt(2)*std(x,0,1)./Y;
H = zeros(length(luminances),length(contrasts));
vacios=H;
for l = 1:length(luminances)
    for m = 1:length(contrasts)
        feature getpid
        [l m]
        ind_lum = find( abs(Y - luminances(l)) <= luminances(2)-luminances(1) );
        ind_con = find( abs(C - contrasts(m)) <= contrasts(2)-contrasts(1) );
        indices_l_m = intersect(ind_lum,ind_con);
        if isempty(indices_l_m)~=1
           samples(l,m).x = x(:,indices_l_m);
           H(l,m) = length(indices_l_m);
        else
           vacios(l,m) = 1; 
        end
    end
end

close all
try
figure,mesh(contrasts,luminances,sqrt(H)),ylabel('L'),xlabel('C'),colormap([0 0 0])
pause(1)
end

%% Multi-Information reduction according to Studeny 98 and theoretical jacobian 

E_log_detJ = zeros(length(luminances),length(contrasts));
E_vol_JND = zeros(length(luminances),length(contrasts));
Delta_H_local = zeros(length(luminances),length(contrasts));

E_log_detJwc = zeros(length(luminances),length(contrasts));
E_vol_JNDwc = zeros(length(luminances),length(contrasts));
Delta_H_localwc = zeros(length(luminances),length(contrasts));
%samples_transf_1 = samples;
%samples_transf_2 = samples;

for l = 1:length(luminances)
    for m = 1:length(contrasts)
        feature getpid
        [l m]
        tic
        if H(l,m) > Nfast
            
           xx = samples(l,m).x;
           xx = xx(:,1:Nfast);
           J = zeros(3,3,length(xx(1,:)));
           JJ = zeros(3,3,length(xx(1,:)));
           y2 = zeros(3,length(xx(1,:)));
           x2 = y2;
           y1wc = zeros(3,length(xx(1,:)));
           y2wc = zeros(3,length(xx(1,:)));
           x2wc = y2wc;
           detJ = zeros(1,length(xx(1,:)));
           volJND = detJ;
           for jj=1:length(xx(1,:))
                [y1,x1,y2(:,jj),x2(:,jj),J1,J2] = stabilized_DN_param(xx(:,jj),param,1);
                J(:,:,jj)=(J2.sx)*(J1.sx);
                detJ(jj) = abs(det(squeeze(J(:,:,jj))));
                volJND(jj) = abs(det( squeeze(J(:,:,jj))'*squeeze(J(:,:,jj)) ));
                % ---------------------------- integracion WC
                % jj
                [y1o,x1o,y2o,x2o,mean_update,mean_MSE,y1wc(:,jj),X1,y2wc(:,jj),x2wc(:,jj),J1o,L2,Jo] = integrate_small_WC( xx(:,jj), param, x2_a, g, k, deltat, Nit);
                
                JJ(:,:,jj) = Jo*L2*(J1o.sx);
                detJwc(jj) = abs(det(squeeze(JJ(:,:,jj))));
                volJNDwc(jj) = abs(det( squeeze(JJ(:,:,jj))'*squeeze(JJ(:,:,jj)) ));
                
                % ----------------------------
           end  

            nbins = round(sqrt(length(xx(1,:))));
            [p01,x01] = hist(xx(1,:),nbins);
            [p02,x02] = hist(xx(2,:),nbins);
            [p03,x03] = hist(xx(3,:),nbins);

            [p21,x21] = hist(x2(1,:),nbins);
            [p22,x22] = hist(x2(2,:),nbins);
            [p23,x23] = hist(x2(3,:),nbins);           

            delta01 = x01(2)-x01(1);
            delta02 = x02(2)-x02(1);
            delta03 = x03(2)-x03(1);
            delta21 = x21(2)-x21(1);
            delta22 = x22(2)-x22(1);
            delta23 = x23(2)-x23(1);

            Delta_H_local(l,m) = entropy_mm(p01)+log2(delta01)+entropy_mm(p02)+log2(delta02)+entropy_mm(p03)+log2(delta03) - entropy_mm(p21)-log2(delta21)-entropy_mm(p22)-log2(delta22)-entropy_mm(p23)-log2(delta23);
           
           samples_transf_1(l,m).y2 = y2;
           samples_transf_2(l,m).x2 = x2;
           clear y2 x2
           jacobians(l,m).J = J;
           E_log_detJ(l,m) = mean(log2(detJ));
           E_vol_JND(l,m) = mean(volJND);
           
           %%%%%% -------------------- guarda inversas de WC y calcula histogramas y medias de jacobianos
           
            nbins = round(sqrt(length(xx(1,:))));
            [p01,x01] = hist(y1wc(1,:),nbins);
            [p02,x02] = hist(y1wc(2,:),nbins);
            [p03,x03] = hist(y1wc(3,:),nbins);

            [p21,x21] = hist(x2wc(1,:),nbins);
            [p22,x22] = hist(x2wc(2,:),nbins);
            [p23,x23] = hist(x2wc(3,:),nbins);           

            delta01 = x01(2)-x01(1);
            delta02 = x02(2)-x02(1);
            delta03 = x03(2)-x03(1);
            delta21 = x21(2)-x21(1);
            delta22 = x22(2)-x22(1);
            delta23 = x23(2)-x23(1);

            Delta_H_localwc(l,m) = entropy_mm(p01)+log2(delta01)+entropy_mm(p02)+log2(delta02)+entropy_mm(p03)+log2(delta03) - entropy_mm(p21)-log2(delta21)-entropy_mm(p22)-log2(delta22)-entropy_mm(p23)-log2(delta23);
           
           samples_transf_0wc(l,m).y1 = y1wc; 
           samples_transf_1wc(l,m).y2 = y2wc;
           samples_transf_2wc(l,m).x2 = x2wc;
           clear y1wc y2wc x2wc
           jacobianswc(l,m).J = JJ;
           E_log_detJwc(l,m) = mean(log2(detJwc));
           E_vol_JNDwc(l,m) = mean(volJNDwc);
        
        elseif (H(l,m) > 200) & (H(l,m) < Nfast)   

           xx = samples(l,m).x;
           J = zeros(3,3,length(xx(1,:)));
           JJ = zeros(3,3,length(xx(1,:)));
           y2 = zeros(3,length(xx(1,:)));
           x2 = y2;
           y1wc = zeros(3,length(xx(1,:)));
           y2wc = zeros(3,length(xx(1,:)));
           x2wc = y2wc;
           detJ = zeros(1,length(xx(1,:)));
           volJND = detJ;
           for jj=1:length(xx(1,:))
                [y1,x1,y2(:,jj),x2(:,jj),J1,J2] = stabilized_DN_param(xx(:,jj),param,1);
                J(:,:,jj)=(J2.sx)*(J1.sx);
                detJ(jj) = abs(det(squeeze(J(:,:,jj))));
                volJND(jj) = abs(det( squeeze(J(:,:,jj))'*squeeze(J(:,:,jj)) ));
                % ---------------------------- integracion WC
                % jj
                [y1o,x1o,y2o,x2o,mean_update,mean_MSE,y1wc(:,jj),X1,y2wc(:,jj),x2wc(:,jj),J1o,L2,Jo] = integrate_small_WC( xx(:,jj), param, x2_a, g, k, deltat, Nit);
                
                JJ(:,:,jj) = Jo*L2*(J1o.sx);
                detJwc(jj) = abs(det(squeeze(JJ(:,:,jj))));
                volJNDwc(jj) = abs(det( squeeze(JJ(:,:,jj))'*squeeze(JJ(:,:,jj)) ));
                
                % ----------------------------
           end  

            nbins = round(sqrt(length(xx(1,:))));
            [p01,x01] = hist(xx(1,:),nbins);
            [p02,x02] = hist(xx(2,:),nbins);
            [p03,x03] = hist(xx(3,:),nbins);

            [p21,x21] = hist(x2(1,:),nbins);
            [p22,x22] = hist(x2(2,:),nbins);
            [p23,x23] = hist(x2(3,:),nbins);           

            delta01 = x01(2)-x01(1);
            delta02 = x02(2)-x02(1);
            delta03 = x03(2)-x03(1);
            delta21 = x21(2)-x21(1);
            delta22 = x22(2)-x22(1);
            delta23 = x23(2)-x23(1);

            Delta_H_local(l,m) = entropy_mm(p01)+log2(delta01)+entropy_mm(p02)+log2(delta02)+entropy_mm(p03)+log2(delta03) - entropy_mm(p21)-log2(delta21)-entropy_mm(p22)-log2(delta22)-entropy_mm(p23)-log2(delta23);
           
           samples_transf_1(l,m).y2 = y2;
           samples_transf_2(l,m).x2 = x2;
           clear y2 x2
           jacobians(l,m).J = J;
           E_log_detJ(l,m) = mean(log2(detJ));
           E_vol_JND(l,m) = mean(volJND);
           
           %%%%%% -------------------- guarda inversas de WC y calcula histogramas y medias de jacobianos
           
            nbins = round(sqrt(length(xx(1,:))));
            [p01,x01] = hist(y1wc(1,:),nbins);
            [p02,x02] = hist(y1wc(2,:),nbins);
            [p03,x03] = hist(y1wc(3,:),nbins);

            [p21,x21] = hist(x2wc(1,:),nbins);
            [p22,x22] = hist(x2wc(2,:),nbins);
            [p23,x23] = hist(x2wc(3,:),nbins);           

            delta01 = x01(2)-x01(1);
            delta02 = x02(2)-x02(1);
            delta03 = x03(2)-x03(1);
            delta21 = x21(2)-x21(1);
            delta22 = x22(2)-x22(1);
            delta23 = x23(2)-x23(1);

            Delta_H_localwc(l,m) = entropy_mm(p01)+log2(delta01)+entropy_mm(p02)+log2(delta02)+entropy_mm(p03)+log2(delta03) - entropy_mm(p21)-log2(delta21)-entropy_mm(p22)-log2(delta22)-entropy_mm(p23)-log2(delta23);
           
           samples_transf_0wc(l,m).y1 = y1wc; 
           samples_transf_1wc(l,m).y2 = y2wc;
           samples_transf_2wc(l,m).x2 = x2wc;
           clear y1wc y2wc x2wc
           jacobianswc(l,m).J = JJ;
           E_log_detJwc(l,m) = mean(log2(detJwc));
           E_vol_JNDwc(l,m) = mean(volJNDwc);
            
        end
        toc
    end
end

Delta_I_local = (Delta_H_local + (E_log_detJ)).*(E_log_detJ~=0) + min((Delta_H_local(:) + (E_log_detJ(:))).*(E_log_detJ(:)~=0)).*(E_log_detJ==0);
Delta_I_localwc = (Delta_H_localwc + (E_log_detJwc)).*(E_log_detJwc~=0) + min((Delta_H_localwc(:) + (E_log_detJwc(:))).*(E_log_detJwc(:)~=0)).*(E_log_detJwc==0);

try
figure,mesh(contrasts,luminances,Delta_I_local),title('\Delta I (DN-Studeny)')
ylabel('L'),xlabel('C'),colormap([0 0 0])

figure,mesh(contrasts,luminances,Delta_I_localwc),title('\Delta I (WC-Studeny)')
ylabel('L'),xlabel('C'),colormap([0 0 0])

figure,mesh(contrasts,luminances,E_vol_JND),title('Sensitivity DN')
ylabel('L'),xlabel('C'),colormap([0 0 0])

figure,mesh(contrasts,luminances,E_vol_JNDwc),title('Sensitivity WC')
ylabel('L'),xlabel('C'),colormap([0 0 0])
end

samples_transform_0 = [];
samples_transform_1 = [];
samples_transform_2 = [];
samples_transform_0wc = [];
samples_transform_1wc = [];
samples_transform_2wc = [];

for l = 1:length(luminances)
    for m = 1:length(contrasts)
        if H(l,m) > 200
           samples_transform_0 = [samples_transform_0 samples(l,m).x]; 
           samples_transform_1 = [samples_transform_1 samples_transf_1(l,m).y2];
           samples_transform_2 = [samples_transform_2 samples_transf_2(l,m).x2];
           
           samples_transform_0wc = [samples_transform_0wc samples_transf_0wc(l,m).y1]; 
           samples_transform_1wc = [samples_transform_1wc samples_transf_1wc(l,m).y2];
           samples_transform_2wc = [samples_transform_2wc samples_transf_2wc(l,m).x2];
        end
    end
end

%% DN
size(samples_transform_0)
nbins = round(sqrt(length(samples_transform_0(1,:))));
[py11,lumin1] = hist(samples_transform_0(1,:),nbins);
[py12,lumin2] = hist(samples_transform_0(2,:),nbins);
[py13,lumin3] = hist(samples_transform_0(3,:),nbins);

[px21,bright2NL] = hist(samples_transform_2(1,:),nbins);
[px22,c_lowNL] = hist(samples_transform_2(2,:),nbins);
[px23,c_highNL] = hist(samples_transform_2(3,:),nbins);

try
figure,plot(lumin1,py11,'r-',lumin2,py12,'g-',lumin3,py13,'b-')
figure,plot(bright2NL,px21,'r-',c_lowNL,px22,'g-',c_highNL,px23,'b-')
end

delta01=lumin1(2)-lumin1(1);
delta02=lumin2(2)-lumin2(1);
delta03=lumin3(2)-lumin3(1);
delta21=bright2NL(2)-bright2NL(1);
delta22=c_lowNL(2)-c_lowNL(1);
delta23=c_highNL(2)-c_highNL(1);

Delta_H_global = entropy_mm(py11)+log2(delta01)+entropy_mm(py12)+log2(delta02)+entropy_mm(py13)+log2(delta03)-entropy_mm(px21)-log2(delta21)-entropy_mm(px22)-log2(delta22)-entropy_mm(px23)-log2(delta23);
Delta_I_global = (Delta_H_global + (E_log_detJ)).*(E_log_detJ~=0) + min((Delta_H_global + (E_log_detJ(:))).*(E_log_detJ(:)~=0)).*(E_log_detJ==0);

try
figure,mesh(contrasts,luminances,Delta_I_global),title('\Delta I (Studeny-DN with \Delta H_{global})')
ylabel('L'),xlabel('C'),colormap([0 0 0])
end

%% WC
nbins = round(sqrt(length(samples_transform_0wc(1,:))));
[py11,lumin1] = hist(samples_transform_0wc(1,:),nbins);
[py12,lumin2] = hist(samples_transform_0wc(2,:),nbins);
[py13,lumin3] = hist(samples_transform_0wc(3,:),nbins);

[px21,bright2NL] = hist(samples_transform_2wc(1,:),nbins);
[px22,c_lowNL] = hist(samples_transform_2wc(2,:),nbins);
[px23,c_highNL] = hist(samples_transform_2wc(3,:),nbins);

try
figure,plot(lumin1,py11,'r-',lumin2,py12,'g-',lumin3,py13,'b-')
figure,plot(bright2NL,px21,'r-',c_lowNL,px22,'g-',c_highNL,px23,'b-')
end

delta01=lumin1(2)-lumin1(1);
delta02=lumin2(2)-lumin2(1);
delta03=lumin3(2)-lumin3(1);
delta21=bright2NL(2)-bright2NL(1);
delta22=c_lowNL(2)-c_lowNL(1);
delta23=c_highNL(2)-c_highNL(1);

Delta_H_globalwc = entropy_mm(py11)+log2(delta01)+entropy_mm(py12)+log2(delta02)+entropy_mm(py13)+log2(delta03)-entropy_mm(px21)-log2(delta21)-entropy_mm(px22)-log2(delta22)-entropy_mm(px23)-log2(delta23);
Delta_I_globalwc = (Delta_H_globalwc + (E_log_detJwc)).*(E_log_detJwc~=0) + min((Delta_H_globalwc + (E_log_detJwc(:))).*(E_log_detJwc(:)~=0)).*(E_log_detJwc==0);

try
figure,mesh(contrasts,luminances,Delta_I_globalwc),title('\Delta I (Studeny-WC with \Delta H_{global})')
ylabel('L'),xlabel('C'),colormap([0 0 0])
end

save(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/3D_example_jesus/redundancy_reduction_small',name],'-v7.3')

%% Multi-Information Reduction using RBIG

DELTA_I = zeros(length(luminances),length(contrasts));

for l = 1:length(luminances)
    for m = 1:length(contrasts)
        if H(l,m) > 200
           tic
           xx = samples(l,m).x; 
           xx = real(xx);
           [gaus1,tans1,parameters1] = RBIG_2017(xx);
           MI1 = parameters1.MI; 
            
           xx2 = samples_transf_2(l,m).x2; 
           xx2 = real(xx2);
           [gaus2,tans2,parameters2] = RBIG_2017(xx2);
           MI2 = parameters2.MI; 

           DELTA_I(l,m) = MI1 - MI2;
           
           feature getpid
           
           [l m]
           toc
           disp('RBIG - DN')
        end
    end
end

DELTA_I_rep = (DELTA_I).*(DELTA_I~=0) + min(DELTA_I(:).*(DELTA_I(:)~=0)).*(DELTA_I==0);

try
figure,mesh(contrasts,luminances,DELTA_I_rep),title('\Delta I - DN (RBIG)')
ylabel('L'),xlabel('C'),colormap([0 0 0]),
end

%% Multi-Information Reduction WC using RBIG

DELTA_Iwc = zeros(length(luminances),length(contrasts));

for l = 1:length(luminances)
    for m = 1:length(contrasts)
        if H(l,m) > 200
           tic
           xx = samples_transf_0wc(l,m).y1; 
           xx = real(xx);
           [gaus1,tans1,parameters1] = RBIG_2017(xx);
           MI1 = parameters1.MI; 
            
           xx2 = samples_transf_2wc(l,m).x2; 
           xx2 = real(xx2);
           [gaus2,tans2,parameters2] = RBIG_2017(xx2);
           MI2 = parameters2.MI; 

           DELTA_Iwc(l,m) = MI1 - MI2;
           feature getpid
           [l m]
           toc
           disp('RBIG - WC')
        end
    end
end

DELTA_I_repwc = (DELTA_Iwc).*(DELTA_Iwc~=0) + min(DELTA_Iwc(:).*(DELTA_Iwc(:)~=0)).*(DELTA_Iwc==0);

try
figure,mesh(contrasts,luminances,DELTA_I_repwc),title('\Delta I - WC (RBIG)')
ylabel('L'),xlabel('C'),colormap([0 0 0]),
end

save(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/3D_example_jesus/redundancy_reduction_small',name],'-v7.3')

% 
% pinta=0;
% 
% if pinta==1
%     
%     %% REDUNDANCY REDUCTION
%     
%     ejes = [0 1 0 1 -5 17];
%     
%     for caso=1:5
%         
%         % save C:\disco_portable\mundo_irreal\latex\Linear_NonLinear\2nd_sumbis_PLOS_Comp_Biol\respuesta\sensitivity_and_redundancy_reduction_base_line
%         load(['/media/disk/vista/Papers/2018_RBIG_IT_measures/redundancy_reduction_case_',num2str(caso)])
%         FIGU = caso*10;
%         color=[1 0.7 0.7;0.7 1 0.7;0.7 0.7 1];
%         
%         % for fre =1:3
%         %
%         % figure(FIGU+1),hold on,colormap(0.6*color(fre,:));h1=mesh(contrasts,luminances,DELTA_H(:,:,fre)),xlabel('C'),ylabel('L'),axis([0 1 0 1 -15 20]),title('\Delta H')
%         % set(h1,'FaceColor',color(fre,:))
%         % box on
%         % figure(FIGU+2),hold on,colormap(0.6*color(fre,:));h2=mesh(contrasts,luminances,E_log_detJ(:,:,fre)),xlabel('C'),ylabel('L'),axis([0 1 0 1 -15 20]),title('E( log_2 |\nabla_x S| )')
%         % set(h2,'FaceColor',color(fre,:))
%         % box on
%         % figure(FIGU+3),hold on,colormap(0.6*color(fre,:));h3=mesh(contrasts,luminances,DELTA_H(:,:,fre) + E_log_detJ(:,:,fre)),xlabel('C'),ylabel('L'),axis([0 1 0 1 -15 20]),title('\Delta I (Studeny 98)')
%         % set(h3,'FaceColor',color(fre,:))
%         % box on
%         % figure(FIGU+4),hold on,colormap(0.6*color(fre,:));h4=mesh(contrasts,luminances,log(E_vol_JND(:,:,fre))),xlabel('C'),ylabel('L'),title('Sensitivity = ( JND Volume )^{-1} = |\nabla_x S^{T} \nabla_x S|')
%         % box on
%         % set(h4,'FaceColor',color(fre,:))
%         % figure(FIGU+5),hold on,colormap(0.6*color(fre,:));h5=mesh(contrasts,luminances,3*DELTA_I(:,:,fre)),xlabel('C'),ylabel('L'),title('\Delta I (RBIG, Laparra 11)')
%         % set(h5,'FaceColor',color(fre,:))
%         % box on
%         %
%         % end
%         %
%         
%         color=[1 0.7 0.7;0.7 1 0.7;0.7 0.7 1];
%         
%         figure(FIGU+6),hold on,colormap(0.6*color(fre,:));h1=mesh(contrasts,luminances,sum(DELTA_H,3)/3),xlabel('C'),ylabel('L'),axis([0 1 0 1 -8 25]),title('\Delta h    (bits)')
%         set(h1,'FaceColor',color(fre,:))
%         box on,view([136 16])
%         set(FIGU+6,'color',[1 1 1]),axis(ejes)
%         figure(FIGU+7),hold on,colormap(0.6*color(fre,:));h2=mesh(contrasts,luminances,sum(E_log_detJ,3)/3),xlabel('C'),ylabel('L'),axis([0 1 0 1 -8 25]),title('E( log_2 |\nabla_x S| )    (bits)')
%         set(h2,'FaceColor',color(fre,:))
%         box on,view([136 16])
%         set(FIGU+7,'color',[1 1 1]),axis(ejes)
%         figure(FIGU+8),hold on,colormap(0.6*color(fre,:));h3=mesh(contrasts,luminances,sum(DELTA_H + E_log_detJ,3)/3),xlabel('C'),ylabel('L'),axis([0 1 0 1 -8 25]),title('\Delta I    (bits)')
%         set(h3,'FaceColor',color(fre,:))
%         box on,view([136 16])
%         set(FIGU+8,'color',[1 1 1]),axis(ejes)
%         
%         figure(FIGU+9),hold on,colormap(0.6*color(fre,:));h4=mesh(contrasts,luminances,log(sum(E_vol_JND,3))/3),xlabel('C'),ylabel('L'),title('Sensitivity = ( JND Volume )^{-1} = |\nabla_x S^{T} \nabla_x S|')
%         box on,view([136 16])
%         set(h4,'FaceColor',color(fre,:))
%         
%         figure(FIGU+10),hold on,colormap(0.6*color(fre,:));h5=mesh(contrasts,luminances,1*sum(DELTA_I(:,:,[1 3]),3)),xlabel('C'),ylabel('L'),axis([0 1 0 1 -8 25]),title('\Delta I (RBIG, Laparra 11)')
%         set(h5,'FaceColor',color(fre,:))
%         box on,view([136 16])
%         
%         
%     end
%     
% end

% figure,mesh(contrasts(1:8),luminances(1:9),3*DELTA_I(1:9,1:8)),
% axis([0 contrasts(8) 0 luminances(9) -7 12]),view([122 24]),title('RBIG efficiency ( \DeltaI )')
% ylabel('L'),xlabel('C'),colormap([0 0 0])
% figure,mesh(contrasts(1:8),luminances(1:9),6+Delta_I(1:9,1:8)),
% axis([0 contrasts(8) 0 luminances(9) -7 12]),view([122 24]),title('Theoret. efficiency ( \DeltaI )')
% ylabel('L'),xlabel('C'),colormap([0 0 0])
