%%
%%  PARAMETERS OF A DIVISIVE NORMALIZATION / WILSON-COWAN MODEL FOR 3-PIXEL IMAGES
%%
%%  We derive sensible parameters for the model in different steps:   
%%
%%  - First we choose sensible parameters for an "unstabilized" Div. Norm. model:
%%          . The brightness nonlinearity
%%          . The spatial transform (a rotation)
%%          . The CSF
%%          . The Gaussian interaction neighborhood
%%          . The exponent of the div. norm.
%%          . The semisaturation of the div. norm.
%%          
%%  - Then we apply the above to natural images to get the stabilization constant, K
%%    (see Front. Neurosci. 18)
%%
%%  - TRICKY: [trial and error] control of left and right high-pass filters in the kernel
%%    so that (in the end, when applid to images) they approx. match K/x and K/b
%%
%%           H = (K/x)*W*(K*f./b)


%%
%%  PARAMETERS AND RESPONSE OF UNSTABILIZED MODEL TO NATURAL IMAGES
%%

FIG = 1;
pinta = 0;

L1 = eye(3);                                                                    
g1= 0.6;
% F2 = Fourier transform (DC,low,high) %% THIS IS ALREADY COMPUTED INSIDE un_stabilized_DN
% F2 = [1 1 1;1 0 -1;-0.5 1 -0.5];
% F2(1,:)=F2(1,:)/norm(F2(1,:));
% F2(2,:)=F2(2,:)/norm(F2(2,:));
% F2(3,:)=F2(3,:)/norm(F2(3,:));
S2 = diag([1 0.5 0.3]); 
g2 = 0.7;
H2 = [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
for i=1:3
    H2(i,:)=H2(i,:)/sum(H2(i,:));  
end
% H2 = eye(3);
b2 = 0.2*[1;0.5;0.4];
b2 = 0.1*[1;0.5;0.4];

for i=1:100
    %c_left = 0.25*[1 2 3]';
    %w_right = 0.5*[1 2 3]';
    %c_left = 0.25*[1 10 100]';
    %w_right = 0.5*[1 10 100]';
    if i == 1
        c_left = [1 2 3]';
        w_right = [1 2 3]';
        c_left = c_left/norm(c_left);
        w_right = w_right/norm(w_right);
    else
        c_left = param(2).K./xm;
        w_right = param(2).K./param(2).b;
        c_left = c_left/norm(c_left);
        w_right = w_right/norm(w_right);
    end
    
    %% Behavior on natural images -> e_star and kappa
    if i==1
        load images_80
        im = [im1 im2 im3]/256;
        x = im2col(im,[1 3],'sliding');
        xx=x;
    end
    
    % x = x(:,1:10000);
    % contrasts = [];
    % luminances = [];
    % for i=1:10000
    %     lum = rand;
    %     C = rand;
    %     contrasts = [contrasts C];
    %     luminances = [luminances lum];
    %     xx(:,i) = control_lum_contrast(x(:,i),lum,C);
    % end
    
    % %% VAN HATEREN
    % load /media/disk/vista/BBDD/Image_Statistic/Van_Hateren/VANH_subsampled_images
    % imag = randperm(4167);
    % num = 50000;
    % xx = [];
    % for i=1:400
    %     im = images(imag(i)).vanH;
    %     x = im2col(im,[1 3],'sliding');
    %     rand_indices=randperm(length(x(1,:)));
    %     xx = [xx x(:,rand_indices(1:round(num/400)))];
    % end
    % clear images
    
    [y1,x1,y2,x2u,J1,J2] = un_stabilized_DN(xx,L1,g1,S2,g2,b2,H2,c_left,w_right,0);
    
    if i == 100
        figure(FIG),
        subplot(131),plot(y2(1,:),x2u(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response (unstab.)')
        subplot(132),plot(y2(2,:),x2u(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
        subplot(133),plot(y2(3,:),x2u(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')
    end
    
    %% Desired scaling
    
    kappa = mean(abs(y2)')';                 % The desired output is the average of the Fourier transform
    kappa = diag([1 1 1])*kappa;       % ad-hoc linear increment of high frequencies 
    % e_star = mean((abs(y2).^g2)')';        % We will have that output for the average energy spectrum
    e_star = kappa.^g2;                      % We will have that output for the energy of the average Fourier transform
    
    %%
    %%  RESPONSE OF STABILIZED TO NATURAL IMAGES
    %%
    
    [y1,x1,y2,x2s,J1,J2,param] = stabilized_DN(xx,L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,0);
    
    % Esta da aprox la misma respuesta (duplicamos el input y duplicamos el output)!!
    % [y1,x1,y2,x2s,J1,J2,param] = stabilized_DN(xx,L1,g1,S2,g2,b2,H2,c_left,w_right,(2^g2)*e_star,2*kappa,0);

    if i==100
    figure,
    subplot(131),plot(y2(1,:),x2s(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response (stab.)')
    subplot(132),plot(y2(2,:),x2s(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
    subplot(133),plot(y2(3,:),x2s(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')
    end
    
    [y1,x1,y2,x2s2,J1,J2] = stabilized_DN_param(xx,param,0);
    
    if i==100
        figure,
        subplot(131),plot(y2(1,:),x2s2(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response (stab.)')
        subplot(132),plot(y2(2,:),x2s2(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
        subplot(133),plot(y2(3,:),x2s2(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')
    end
    
    % In natural images the output for the emplitudes e_star^(1/g) is reduced wrt kappa (a bit) because of the efect of the neighbors
    
    nbins = round(sqrt(length(xx(1,:))));
    [py11,lumin1] = hist(xx(1,:),nbins);
    [py12,lumin2] = hist(xx(2,:),nbins);
    [py13,lumin3] = hist(xx(3,:),nbins);
    
    [px21,bright2NL] = hist(x2s2(1,:),nbins);
    [px22,c_lowNL] = hist(x2s2(2,:),nbins);
    [px23,c_highNL] = hist(x2s2(3,:),nbins);
    
    if i==100
        figure,semilogy(lumin1,py11,'r-',lumin2,py12,'g-',lumin3,py13,'b-')
        figure,semilogy(bright2NL,px21,'r-',c_lowNL,px22,'g-',c_highNL,px23,'b-')

        figure,plot3(xx(1,1:10:end),xx(2,1:10:end),xx(3,1:10:end),'.'),title('luminance')
        figure,plot3(x1(1,1:10:end),x1(2,1:10:end),x1(3,1:10:end),'.'),title('brightness')
        figure,plot3(y2(1,1:10:end),y2(2,1:10:end),y2(3,1:10:end),'.'),title('contrast')
        figure,plot3(x2s(1,1:10:end),x2s(2,1:10:end),x2s(3,1:10:end),'.'),title('nonlinear contrast')
    end

    i
    xm = mean(abs(x2s2'))';
    
    xm'
    param(2).K'
    c_left'
    w_right'
    % pause
end

W = inv(diag(param(2).Hc))*param(2).H*inv(diag(param(2).Hw));
param(2).W = W;
alfa_m = param(2).b./param(2).K;
param(2).alpha = alfa_m;

% %%
% %% CHECK JACOBIAN  
% %%
% 
% x0 = 0.5*[1 1 1]'; 
% x0 = rand(3,1);
% 
% %[x2,NablaS,nabla1,nabla2,K] = stabilized_DN_3d(x0,L1,g1,S2,g2,b2,H2,0,0);
% [y1,x1,y2,x2,J1,J2,param] = stabilized_DN(x0,L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,1);
% 
% NablaS = (J2.sx)*(J1.sx);
% epsilon = 1e-8;               %% bigger step is worse (logic) but extremely small is worse too (round-off errors)
% nabla_num = 0*NablaS;
% for j=1:3
%         delta=zeros(3,1);
%         delta(j)=epsilon;
%         %[x22,NablaS2] = stabilized_DN_3d(x0+delta,L1,g1,S2,g2,b2,H2,0,0);
%         [y1c,x1c,y2c,x2c,J1,J2,parapara] = stabilized_DN(x0+delta,L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,0);
%         %x2c = inv(param(2).K)*x2c;
%         nabla_num(:,j) = (x2c-x2)/epsilon;
% end
% (sum(sum((NablaS - nabla_num).^2)))/sum(sum(NablaS.^2))
% 
% %%
% %%  Viendolo etapa a etapa
% %%
% 
% comp_J.sx=1;
% [y1,x1,J1] = stage_L_NL_c(x0,param(1),comp_J);
% [y2,x2,J2] = stage_L_NL_c(x1,param(2),comp_J);
% x2=diag(param(2).K)*x2;
% 
% NablaS1 = J1.sx;
% NablaS2 = diag(param(2).K)*J2.sx;
% nabla_num1 = 0*NablaS1;
% nabla_num2 = 0*NablaS2;
% nabla_num12 = 0*NablaS2;
% 
% for j=1:3
%     delta=zeros(3,1);
%     delta(j)=epsilon;
%     % [x22,NablaS2] = stabilized_DN_3d(x0+delta,L1,g1,S2,g2,b2,H2,0,0);
%     % [y1c,x1c,y2c,x2c,J1,J2] = stabilized_DN(x0+delta,L1,g1,S2,g2,b2,H2,0,0);
%     [y1m,x1m,J1m] = stage_L_NL_c(x0+delta,param(1),0);
%     [y2m,x2m,J2m] = stage_L_NL_c(x1+delta,param(2),0);
%     x2m=diag(param(2).K)*x2m;
%     nabla_num1(:,j) = (x1m-x1)/epsilon;
%     nabla_num2(:,j) = (x2m-x2)/epsilon;
% 
%     [y2m,x2m,J2m] = stage_L_NL_c(x1m,param(2),0);
%     x2m = diag(param(2).K)*x2m;
%     nabla_num12(:,j) = (x2m-x2)/epsilon;
% end
% 
% (sum(sum((NablaS1-nabla_num1).^2)))/sum(sum((NablaS1).^2))
% (sum(sum((NablaS2-nabla_num2).^2)))/sum(sum((NablaS2).^2))
% (sum(sum(((NablaS2*NablaS1)-nabla_num12).^2)))/sum(sum((NablaS2*NablaS1).^2))
% 
% %%
% %%  STABILITY ANALYSIS 3D
% %%
% 
% x0 = rand(3,1);
% [y1,x1,y2,x2,J1,J2,parame] = stabilized_DN(x0,L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,1);
% b = parame(2).b;
% K = parame(2).K;
% c = parame(2).Hc;
% w = parame(2).Hw;
% H = parame(2).H;
% W = diag(1./c)*H*diag(1./w);
% 
% J_stab_static = -( diag(b./K) + W);
% J_stab_dynamic = -( diag(b./K) + diag(x2./K)*H*diag(b./K) );
% 
% [x0 eig(J_stab_static) eig(J_stab_dynamic)]
% 
% %% Stability aroun a certain point in the Fourier domain
% 
% x_ref_fourier = [0.5;0;0];
% x_ref_0 = (inv(param(2).L)*x_ref_fourier).^(1/g1);
% [y1ref,x1ref,y2ref,x2ref,J1,J2,parame] = stabilized_DN(x_ref_0,L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,1);
% 
% num = 10;
% [X,Y] = meshgrid(linspace(-8*kappa(2),8*kappa(2),num),linspace(-8*kappa(3),8*kappa(3),num));    
% x0 = (inv(param(2).L)*(x_ref_fourier+[0.5*ones(length(X(:)),1)';X(:)';Y(:)']));
% x0 = sign(x0).*abs(x0).^(1/g1);
% x0_fourier = [0.5*ones(length(X(:)),1)';X(:)';Y(:)'];
% 
% delta_x_stat = [];
% delta_x_dynam = [];
% 
% for i=1:length(X(:))
%     
%     [y1,x1,y2,x2,J1,J2,parame] = stabilized_DN(x0(:,i),L1,g1,S2,g2,b2,H2,c_left,w_right,e_star,kappa,1);
%     perturb_x = x2 - x2ref;
%     J_stab_static = -( diag(b./K) + W);
%     J_stab_dynamic = -( diag(b./K) + diag(x2./K)*H*diag(b./K) );
%     
%     delta_x_stat = [delta_x_stat  J_stab_static*perturb_x ];
%     delta_x_dynam = [delta_x_dynam  J_stab_dynamic*perturb_x ];
%     
% end
% 
% delta_x0 = (inv(param(2).L)*(delta_x_stat));
% delta_x0 = sign(delta_x0).*abs(delta_x0).^(1/g1);
% 
% figure,quiver(x0(2,:),x0(3,:),delta_x0(2,:),delta_x0(3,:)),title({'Phase Space of Divisive Norm. [signal indep]';'(stability of Wilson-Cowan steady state)'})
% xlabel('Perturbation in x^{(0)}_2')
% ylabel('Perturbation in x^{(0)}_3')
% 
% figure,quiver(x0_fourier(2,:),x0_fourier(3,:),delta_x_stat(2,:),delta_x_stat(3,:)),title({'Phase Space of Divisive Norm. [signal indep]';'(stability of Wilson-Cowan steady state)'})
% hold on,plot([-8*kappa(2) 8*kappa(2)],[0 0],'k-')
% hold on,plot([0 0],[-8*kappa(3) 8*kappa(3)],'k-')
% xlabel('Perturbation in y^{(2)}_2')
% ylabel('Perturbation in y^{(2)}_3')
% 
% figure,quiver(x0_fourier(2,:),x0_fourier(3,:),delta_x_dynam(2,:),delta_x_dynam(3,:)),title({'Phase Space of Divisive Norm.  [signal dependent]';'(stability of Wilson-Cowan steady state)'})
% hold on,plot([-8*kappa(2) 8*kappa(2)],[0 0],'k-')
% hold on,plot([0 0],[-8*kappa(3) 8*kappa(3)],'k-')
% xlabel('Perturbation in y^{(2)}_2')
% ylabel('Perturbation in y^{(2)}_3')
% 
% %% LA GRAFICA XULA
% 
% figure,
% t=quiver(x0_fourier(2,:),x0_fourier(3,:),delta_x_stat(2,:),delta_x_stat(3,:)),
% set(t,'linewidth',2)
% title({'Phase Space of Divisive Normalization';'(two coefficients in the V1 - Wavelet domain)'})
% hold on,plot([-8*kappa(2) 8*kappa(2)],[0 0],'k-')
% hold on,plot([0 0],[-8*kappa(3) 8*kappa(3)],'k-')
% xlabel('Perturbation in x^{(4)}_i')
% ylabel('Perturbation in x^{(4)}_j')
% set(gcf,'color',[1 1 1])
% 
% %%
% %%  STABILITY ANALYSIS nD
% %%
% 
% fs=64; % sampling frequency in cpd
% %% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment this in workstations)
% % N=64;  % Number of samples -> 64*64 image blocks subtend 1 degree
% % NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment this in personal computers)
%  N=40;  % Number of samples -> 64*64 image blocks subtend 1 degree (40*40 -> patches of less than 1 deg)
% 
% %% PARAMETERS OF THE 4 LAYERS OF THE MODEL
% %% (see description of the parameters in PARAMETERS_DN_ISO_COLOR.M)
% 
% % Layer 1: Brightness and RG-ness and YB-ness
% 
% g1=[1.25 1.25 1.25];  b1 = [0.01 0.01 0.01]; beta = [5 2 2]; mu = [0 0 0]; scale=[255 23 35]; % Weber-Laughlin-Malo-Bertalm
% 
% % Layer 2: Contrast computation (achromatic and chromatic)
% Ls=0.066; Hs=0.066; Lc=1; Hc=1; b2=[30 10*23 10*35];                                          % Gutierrez 12 + singul
% 
% % Layer 3: CSFs and masking in the spatial domain
% obliq=1; lambda_regul_CSF=0.5e-4; b3=[0.005 0.00075 0.00075]; g3=1.5; H3s=0.02; H3c=1;        % Mullen-Barlow-Malo+reduct sing
% 
% % Layer 4: Wavelet transform and masking in the wavelet domain
% 
% %% NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment the following 4 lines in workstations -and comment the case below for small patches-)
% % ns = 4; no=4; tw=1; [p,ind] = buildSFpyr(rand(N,N),ns,no-1,tw); n_sub = length(ind(:,1));
% %w = [50 50*[1 1 1 1] 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.001];                        % Malo-Bertalm-Wilson-Cowan
% %c = [50 50*[1 1 1 1] 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.01];                         % 4 scales * 4 orient + 2 residuals
% %b4 = 0.005*[1*[1 1 1 1 1] 1*[1 1 1 1] 10*[1 1 1 1] 100*[1 1 1 1] 0.8*100/2.5];
% % NOTE: IMAGE SIZE AND SCALES OF THE WAVELET!  (uncomment the following 4 lines in personal computers -and comment the case above for big patches-)
% ns = 3; no=4; tw=1; [p,ind] = buildSFpyr(rand(N,N),ns,no-1,tw); n_sub = length(ind(:,1));
% w = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.001];
% c = [50 50*[1 1 1 1] 0.1*[1 1 1 1] 0.05*[1 1 1 1] 0.01];
% b4 = 0.005*[1*[1 1 1 1 1] 10*[1 1 1 1] 100*[1 1 1 1] 0.8*100/2.5];
% %%---------------
% c_ff_interact = ones(n_sub,n_sub);                                                            
% g4 = 0.6;
% sigma_x = 0.08*ones(ns+2,1);  sigma_x(end)=0.04; sigma_scale = 1.1; sigma_orient = 30;        % Watson-Solomon
% autonorm = 0; save_for_deriv = 0; save_for_inv = 0; save_achrom = 0;
% 
% % Compute the parameters
% tic
% parameters_DN_iso_color      % generates the variable "param" with all the parameters of the model                                                               
% toc
% 
% H = param(4).H;
% b = param(4).b;
% d = param(4).d;
% [Structure_Matrix , b] = structure( param(4).ind , param(4).d , b);
% c = param(4).Hc;
% [Structure_Matrix , c] = structure( param(4).ind , param(4).d , c);
% w = param(4).Hw;
% [Structure_Matrix , w] = structure( param(4).ind , param(4).d , w);
% W = repmat(1./c,[1 d]).*H.*repmat((1./w)',[d 1]);
% 
% e_star = param(4).e_average;
% kappa = param(4).kappa;
% K = kappa(:,1).*(b+H*e_star)./e_star;
% 
% J_stab_static = -( diag(b./K) + W);
% % J_stab_dynamic = -( diag(b./K) + diag(x2./K)*H*diag(b./K) );
% 
% clear H W
% tic,[V,D_lambda]=eig(J_stab_static);toc
% 
% v=diag(D_lambda);
% 
% figure(1),loglog(v,'r-','linewidth',2),
% %title({['Stability of Divisive Normalization'];['Eigenspectrum of Jacobian of rhs of Wilson-Cowan']})
% t1=text(1.5,-0.3*1e-1,['Stability of the Divisive Normalization solution'])
% t2=text(1.5,-0.3*2e-1,['(Eigenspectrum of Jacobian of rhs of Wilson-Cowan)'])
% set(t1,'fontsize',14,'FontWeight','bold')
% set(t2,'fontsize',11,'FontWeight','bold')
% xlabel('Index of Eigenspectrum')
% ylabel('Eigenvalues    (ALL NEGATIVE)')
% set(gcf,'color',[1 1 1])
% 
