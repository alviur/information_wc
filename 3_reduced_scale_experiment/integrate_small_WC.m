function [y1,x1,y2,x2,mean_update,mean_MSE,Y1,X1,YY,Xint,J1,L2,J] = integrate_small_WC(xx,param_DN,xx_a,g,k,deltat,Nit)

% 
% INTEGRATE_SMALL_WC computes the response of a small-scale vision model based on
% two interaction schemes after the frequency-sensitive neurons:
%  - Divisive Normalization interaction
%  - Wilson-Cowan interaction
%
% It requires the parameters of the Divisive Normalization transform
% (computed with parameters_3D_small.m)
%
% [y1,x1,y2,x2,mean_update,mean_MSE,Y1,X1,Y2,X2,J1,L2,J] = integrate_small_WC( y1, param_DN, x2_a, g, k, deltat, Nit);
%


%
%   Intento 6: ------------------------
%                     * csf               [1 0.5 0.3]
%                     * kappa             [1 1 1]
%   In Div. Norm.:    * Width of H        [1 0.06 0.01;0.03 1 0.05;0.001 0.01 1];
%                     * c_left, w_right   [1 2 3]
%                     * b                 0.1*[1 0.5 0.4]  
%
%   In integration:   * exponent          0.4
%                     * k                 [1/8 1/15 1/60].^g  0.75*k/max(k)
% 
%        BIEN!: Las marginales inversas salen ok
%        MAL: Hay mas diferencia entre la escala de DN el altas y la de WC (factor 10)
%        BIEN!: converge
%

%close all
%parameters_3D_small  % -> param
%close(1),close(2),close(4),close(6),close(7),close(8),close(9),%close(10),close(11)

%W = inv(diag(param(2).Hc))*param(2).H*inv(diag(param(2).Hw));
%param(2).W = W;
%alfa_m = param(2).b./param(2).K;
%param(2).alpha = alfa_m;

%load images_80  
%im = [im1 im2 im3]/256;
%x = im2col(im,[1 3],'sliding');
%xx=x;

alfa_m = param_DN(2).alpha;
W = param_DN(2).W;

[y1,x1,y2,x2,J1,J2] = stabilized_DN_param(xx,param_DN,0);

% xx_a = mean(abs(x2)')';
% g = 0.4;
% k = diag([1/8 1/15 1/60].^g);
% k = 0.75*k/max(k(:));
% deltat = 1e-5;
% DmW = diag(alfa_m) + W;

Da = diag(alfa_m);
% WDxm = W*diag(xx_a.^(1-g));
    
L = length(xx(1,:));
DIF = [];
DIF_x = [];
Xint = [];
X = [];
Y = [];
D = [];
YY = [];
J = zeros(3,3,L);

for una_imagen = 1:L

    % e = abs(y2(:,una_imagen)).^param(2).g;
    e = abs(y2(:,una_imagen)).^g;
    x = abs(x2(:,una_imagen));
    Y = [Y y2(:,una_imagen)];
    X = [X sign(y2(:,una_imagen)).*x];
    
    xt = e;
    
    %% 
    %% Euler Integration
    %% 
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    xs = [];
    % tic
    for t=1:Nit
        % Usando la no-linealidad simplificada
        % xtm1 = xt + e*deltat - k*Da*xt*deltat - k*WDxm*(sign(xt).*abs(xt).^g)*deltat;
        
        % Usando la no-linealidad no-singular
        [f,dfdx] = saturation_f(xt,g,xx_a,0.1*xx_a);
        xtm1 = xt + e*deltat - k*Da*xt*deltat - k*W*f*deltat;
        xt = xtm1;
        %  figure(100),plot(xtm1);axis([0 length(e) 0 max(e)]);title(num2str(t))
        xs = [xs xt];
        % if mod(t,500)==0
        %   [una_imagen t]
        % end
    end
    % e_time =toc;  
    if mod(una_imagen,500)==0
       una_imagen
    end    
    dif = xs(:,2:end)-xs(:,1:end-1);
    dif_x = xs - repmat(x,[1 length(xs(1,:))]);
    
    DIF = [DIF;mean(abs(dif).^2)/mean(abs(x).^2)];
    DIF_x = [DIF_x;mean(abs(dif_x).^2)/mean(abs(x).^2)];
    Xint = [Xint sign(y2(:,una_imagen)).*xs(:,end)];
    D = [D xs(:,end)-x]; 
    
    % y = sign(y2(:,una_imagen)).*( k*(Da*xs(:,end) + WDxm*(abs(xs(:,end)).^g)) ).^(1/g);
    
    [f,dfdx] = saturation_f(xs(:,end),g,xx_a,0.1*xx_a);
    y = sign(y2(:,una_imagen)).*( k*(Da*xs(:,end) + W*f) ).^(1/g);
    
    YY = [YY y];
    
    %JJ = g*inv(Da + fact*WW_full_2ml.*repmat(dfdx',[size(WW_full_2ml,1) 1])).*repmat(((Da*abs(samples_2_ml_x4wc(:,i)) + fact*WW_full_2ml*f).^(1-(1/g)))',[size(WW_full_2ml,1) 1] );
    JJ = g*inv(k*Da + k*W*diag(dfdx))*diag( (k*Da*abs(xs(:,end)) + k*W*f).^(1-(1/g)) );
    % Since numbers are so small det is close to zero and log gives -inf ->
    % take eigenvalues and sum logs instead of log(product)
    % lambda=abs(eig(J));
    % average = average + sum(log2(lambda));
    
    J(:,:,una_imagen) = JJ;
    
end

%figure(11),plot3(y1(1,:),y1(2,:),y1(3,:),'.'),title('Luminance')
%figure(12),plot3(x1(1,:),x1(2,:),x1(3,:),'.'),title('Brightness')

% Xint
% YY
X1 = param_DN(2).iL*YY;
Y1 = sign(X1).*(abs(X1).^(1/param_DN(1).g));

computJ.sx = 1;
[yy,xim1,J1]=stage_L_NL_c(Y1,param_DN(1),computJ);
L2 = param_DN(2).L;

mean_update = mean(abs(DIF));
mean_MSE = mean(abs(DIF_x));

%figure(13),plot3(Y1(1,:),Y1(2,:),Y1(3,:),'.'),title('Luminance (inverted from WC)')
%figure(14),plot3(X1(1,:),X1(2,:),X1(3,:),'.'),title('Brightness (inverted from WC)')

%figure(15),plot3(Y(1,:),Y(2,:),Y(3,:),'.'),title('Contrast')
%figure(16),plot3(YY(1,:),YY(2,:),YY(3,:),'.'),title('Contrast (inverting WC)')

%figure(17),plot3(X(1,:),X(2,:),X(3,:),'.'),title('DN')
%figure(18),plot3(Xint(1,:),Xint(2,:),Xint(3,:),'.'),title('WC')

%figure(19),loglog(mean(abs(DIF))),title('Relative energy of update'),
%figure(20),loglog(mean(abs(DIF_x))),title('Relative MSE from DN'),

%figure(21),
%subplot(131),plot(Y(1,:),Xint(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response WC')
%subplot(132),plot(Y(2,:),Xint(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
%subplot(133),plot(Y(3,:),Xint(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')

%figure(210),
%subplot(131),plot(YY(1,:),Xint(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response WC')
%subplot(132),plot(YY(2,:),Xint(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
%subplot(133),plot(YY(3,:),Xint(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')

%figure(211),
%subplot(131),plot(Y(1,:),X(1,:),'r.'),title('Zero-frequency'),xlabel('Ampl.1 (Brightness)'),ylabel('Response DN')
%subplot(132),plot(Y(2,:),X(2,:),'g.'),title('Low-frequency'),xlabel('Ampl.2')
%subplot(133),plot(Y(3,:),X(3,:),'b.'),title('High-frequency'),xlabel('Ampl.3')

% 
% [py21,bright2L] = hist(Y(1,:),0.4*nbins);
% [py22,c_lowL] = hist(Y(2,:),0.4*nbins);
% [py23,c_highL] = hist(Y(3,:),0.4*nbins);
% figure(22),semilogy(bright2L,py21,'r-',c_lowL,py22,'g-',c_highL,py23,'b-'),title('PDFs Linear')
% 
% [py21i,bright2Li] = hist(YY(1,:),0.4*nbins);
% [py22i,c_lowLi] = hist(YY(2,:),0.4*nbins);
% [py23i,c_highLi] = hist(YY(3,:),0.4*nbins);
% figure(23),semilogy(bright2Li,py21i,'r-',c_lowLi,py22i,'g-',c_highLi,py23i,'b-'),title('PDFs Linear (inverted)')
% 
% [px21,bright2NL] = hist(Xint(1,:),0.4*nbins);
% [px22,c_lowNL] = hist(Xint(2,:),0.4*nbins);
% [px23,c_highNL] = hist(Xint(3,:),0.4*nbins);
% figure(24),semilogy(bright2NL,px21,'r-',c_lowNL,px22,'g-',c_highNL,px23,'b-'),title('PDFs WC')
